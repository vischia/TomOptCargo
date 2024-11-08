
from typing import List, Union, Optional, Iterator, Tuple
import torch
from torch import Tensor, nn
import numpy as np

from volume.hodoscope import Hodoscope

from tomopt.core import DEVICE
from tomopt.volume.layer import AbsDetectorLayer
from tomopt.volume.panel import DetectorPanel
from tomopt.muon import MuonBatch

r"""
Provides implementations of active detection layers containing hodoscope-like detectors.
"""

class HodoscopeDetectorLayer(AbsDetectorLayer):
    def __init__(
        self,
        pos: str,
        *,
        lw: Tensor,
        z: float,
        size: float,
        hodoscopes: List[Hodoscope],
    ):
        if isinstance(hodoscopes, list):
            hodoscopes = nn.ModuleList(hodoscopes)

        super().__init__(
            pos=pos, lw=lw, z=z, size=size, device=self.get_device(hodoscopes)
        )
        self.hodoscopes = hodoscopes

        if isinstance(hodoscopes[0], Hodoscope):
            self.type_label = "hodoscope"
            self._n_costs = len(self.hodoscopes)
        else:
            raise TypeError("Provided hodoscopes have type {} and not Hodoscope.".format(type(hodoscopes[0])))

    @staticmethod
    def get_device(hodoscopes: nn.ModuleList) -> torch.device:
        r"""
        Helper method to ensure that all hodoscopes are on the same device, and return that device.
        If not all the hodoscopes are on the same device, then an exception will be raised.

        Arguments:
            hodoscopes: ModuleLists :class:`Hodoscope` objects on device

        Returns:
            Device on which all the hodoscopes are.
        """

        device = hodoscopes[0].device
        if len(hodoscopes) > 1:
            for h in hodoscopes[1:]:
                if h.device != device:
                    raise ValueError(
                        "All hodoscopes must use the same device, but found multiple devices"
                    )
        return device

    def get_panel_zorder(self) -> List[int]:
        r"""
        Returns:
            The indices of the panels in order of decreasing z-position.
        """

        return list(
            np.argsort(
                [p.z.detach().cpu().item() for h in self.hodoscopes for p in h.panels]
            )[::-1]
        )

    def yield_zordered_panels(self) -> Iterator[Tuple[int, DetectorPanel]]:
        r"""
        Yields the index of the panel, and the panel, in order of decreasing z-position.

        Returns:
            Iterator yielding panel indices and panels in order of decreasing z-position.
        """
        panels = [p for h in self.hodoscopes for p in h.panels]

        for i in self.get_panel_zorder():
            yield i, panels[i]

    def forward(self, mu: MuonBatch) -> None:
        r"""
        Propagates muons to each detector panel, in order of decreasing z-position, and calls their `get_hits` method to record hits to the muon batch.
        After this, the muons will be propagated to the bottom of the detector layer.

        Arguments:
            mu: the incoming batch of muons
        """

        for i, p in self.yield_zordered_panels():
            mu.propagate_dz(mu.z - p.z.detach())  # Move to panel
            hits = p.get_hits(mu)
            mu.append_hits(hits, self.pos)
        mu.propagate_dz(mu.z - (self.z - self.size))  # Move to bottom of layer

    def conform_detector(self) -> None:
        r"""
        Loops through hodoscopes and calls their `clamp_params` method, to ensure that hodoscopes are located within the bounds allocated to them within the detector layer.
        Assuming hodoscopes are created along the x direction, the total length of the layer is split among them, to prevent any overlap in x. 
        Hodoscopes are free to move in y and z directions within the layer. 
        It will be called via the :class:`~tomopt.optimisation.wrapper.AbsVolumeWrapper` after any update to the hodoscope layers.
        """

        lw = self.lw.detach().cpu().numpy()
        z = self.z.detach().cpu()[0]
        n= len(self.hodoscopes)
        for i,p in enumerate(self.hodoscopes):
            x_low = 0 if i==0 else i*lw[0]/n + p.xyz_span[0].detach().cpu().numpy()/2
            x_high = lw[0]/n - p.xyz_span[0].detach().cpu().numpy()/2 if i==0 else (i+1)*lw[0]/n
            p.clamp_params(
                xyz_low= (x_low , 0, z - self.size + p.xyz_span[2]), # added thickness of hodoscope to lower z value to ensure it stays inside layer
                xyz_high=(x_high, lw[1], z),

            )

    def get_cost(self) -> Tensor:
        r"""
        Returns the total, current cost of the detector(s) in the layer, as computed by looping over the hodoscopes and summing the returned values of calls to
        their `get_cost` methods.

        Returns:
            Single-element tensor with the current total cost of the detector in the layer.
        """

        cost = None
        panels = [p for h in self.hodoscopes for p in h.panels]
        for p in panels:
            cost = p.get_cost() if cost is None else cost + p.get_cost()
