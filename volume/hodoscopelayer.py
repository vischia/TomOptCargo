from tomopt.volume.layer import AbsDetectorLayer
from tomopt.volume.panel import DetectorPanel
from typing import List, Union, Optional, Iterator, Tuple
import torch
from torch import Tensor
import numpy as np
from tomopt.muon import MuonBatch
from volume.hodoscope import Hodoscope
from torch import nn


from tomopt.core import DEVICE


class HodoscopeDetectorLayer(AbsDetectorLayer):

    def __init__(self, 
                 pos:str, 
                 *,
                 lw:Tensor,
                 z:float,
                 size:float, 
                 hodoscopes: List[Hodoscope],
    ):
        if isinstance(hodoscopes, list):
            hodoscopes = nn.ModuleList(hodoscopes)

        super().__init__(pos=pos, lw=lw, z=z, size=size, device=self.get_device(hodoscopes))
        self.hodoscopes = hodoscopes

        if isinstance(hodoscopes[0], Hodoscope):
            self.type_label = "hodoscope"
            self._n_costs = len(self.hodoscopes)

    @staticmethod
    def get_device(hodoscopes: nn.ModuleList) -> torch.device:

        r"""
        Helper method to ensure that all panels are on the same device, and return that device.
        If not all the panels are on the same device, then an exception will be raised.

        Arguments:
            panels: ModuleLists of either :class:`~tomopt.volume.panel.DetectorPanel` or :class:`~tomopt.volume.heatmap.DetectorHeatMap` objects on device

        Returns:
            Device on which all the panels are.
        """

        device = hodoscopes[0].device
        if len(hodoscopes) > 1:
            for h in hodoscopes[1:]:
                if h.device != device:
                    raise ValueError("All hodoscopes must use the same device, but found multiple devices")
        return device
    
    def get_panel_zorder(self) -> List[int]:
        r"""
        Returns:
            The indices of the panels in order of decreasing z-position.
        """

        return list(np.argsort([p.z.detach().cpu().item() for h in self.hodoscopes for p in h.panels])[::-1])
    
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
        Loops through hodoscopes and calls their `clamp_params` method, to ensure that hodoscopes are located within the bounds of the detector layer.
        It will be called via the :class:`~tomopt.optimisation.wrapper.AbsVolumeWrapper` after any update to the hodoscope layers.
        """

        lw = self.lw.detach().cpu().numpy()
        z = self.z.detach().cpu()[0]
        for p in self.hodoscopes:
            p.clamp_params(
                xyz_low=(0, 0, z - self.size),
                xyz_high=(lw[0], lw[1], z),
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
