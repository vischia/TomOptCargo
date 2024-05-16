from typing import Tuple, List, Optional, Union
import numpy as np
import torch
from torch import Tensor, nn

from volume.panel import HodoscopeDetectorPanel, SigmoidHodoscopeDetectorPanel

from tomopt.core import DEVICE

r"""
Provides implementations of class simulating hodoscope-style detectors with learnable xyz positions.
"""

__all__ = ["Hodoscope"]


class Hodoscope(nn.Module):
    r"""
    Provides a Hodoscope detection system containing indepedent detction panels, 
    centered at a learnable parameter `xyz` position and with a fixed span `xy_span`.
    Whilst this class can be used manually, it is designed to be used by the :class:`~tomopt.volume.layer.PanelDetectorLayer` class.
    
    Once istanciated, the Hodoscope class creates HodoscopedetectorPanels centered at `xy`. They are responsible for the muon hit recording throught their `get_hits` method.
    
    Arguments:
        init_xyz: initial xyz position of the hodoscope in metres in the volume frame. The z position corresponds to the top of the hodoscope.
        init_xyz_span: initial xyz-span (total width) of the hodoscope in metres.
        xyz_gap: the gap between the edges of the hodoscope and the panels.
        n_panels: the number of detection panels within the hodoscope.
        res: resolution of the panels in m^-1, i.e. a higher value improves the precision on the hit recording.
        eff: efficiency of the hit recording of the panel, indicated as a probability [0,1].
        realistic_validation: if True, will use the physical interpretation of the panel during evaluation.
        device: device on which to place tensors.
    """

    def __init__(self,
                 *,
                 init_xyz: Tuple[float, float, float],
                 init_xyz_span: Tuple[float, float, float],
                 xyz_gap: Tuple[float, float, float],
                 n_panels: int = 3,
                 res: float = 1000,
                 eff: float = 0.9,
                 realistic_validation: bool = False,
                 panel_type: str = 'DetectorPanel',
                 smooth: Union[float, Tensor] = None,
                 device: torch.device = DEVICE):
        
        if res <= 0:
            raise ValueError("Resolution must be positive")
        if eff <= 0:
            raise ValueError("Efficiency must be positive") 

        super().__init__()
        self.realistic_validation, self.device = realistic_validation, device
        self.xy = nn.Parameter(torch.tensor(init_xyz[:2], device=self.device))
        self.z = nn.Parameter(torch.tensor(init_xyz[2:3], device=self.device))
        # self.init_z = init_xyz[-1]
        self.xyz_span = torch.Tensor(init_xyz_span, device=self.device)
        self.register_buffer("resolution", torch.tensor(float(res), device=self.device))
        self.register_buffer("efficiency", torch.tensor(float(eff), device=self.device))
        self.xyz_gap = xyz_gap
        self.n_panels = n_panels
        self.panel_type = panel_type
        self.smooth = smooth
        self.panels = self.generate_init_panels()
        self.device = device

    
    def __getitem__(self, idx: int) -> HodoscopeDetectorPanel:
        return self.panels[idx]

    def generate_init_panels(self) -> List[HodoscopeDetectorPanel]:
        r"""
        Generates Detector panels based on the xy and z position (xy, z), the span of the hodoscope (xyz_span), 
        and the gap between the edge of the hodoscope and the panels (xyz_gap).

        Returns:
            DetectorPanels as a nn.ModuleList.
        """
        if self.panel_type == 'DetectorPanel':
            return [HodoscopeDetectorPanel(realistic_validation = self.realistic_validation,
                                            idx = i, 
                                            init_xy_span = [self.xyz_span[0] - 2 * self.xyz_gap[0], self.xyz_span[1] - 2 * self.xyz_gap[1]],
                                            device = DEVICE, 
                                            hod = self) for i in range(self.n_panels)]
                
        elif self.panel_type == 'SigmoidDetectorPanel':
            return [SigmoidHodoscopeDetectorPanel(smooth = self.smooth,
                                            realistic_validation = self.realistic_validation,
                                            idx = i, 
                                            init_xy_span = [self.xyz_span[0] - 2 * self.xyz_gap[0], self.xyz_span[1] - 2 * self.xyz_gap[1]],
                                            device = DEVICE, 
                                            hod = self) for i in range(self.n_panels)]
        
        else:
            raise ValueError(f"Detector type {self.panel_type} currently not supported.")

    def clamp_params(self, xyz_low: Tuple[float, float, float], xyz_high: Tuple[float, float, float]) -> None:
        r"""
        Ensures that the panel is centred within the supplied xyz range.
        A small random number < 1e-3 is added/subtracted to the min/max z position of the panel, to ensure it doesn't overlap with other panels.

        Arguments:
            xyz_low: minimum x,y,z values for the panel centre in metres
            xyz_high: maximum x,y,z values for the panel centre in metres
        """

        with torch.no_grad():
            eps = np.random.uniform(0, 1e-3)  # prevent hits at same z due to clamping
            self.xy[0].clamp_(min=xyz_low[0], max=xyz_high[0])
            self.xy[1].clamp_(min=xyz_low[1], max=xyz_high[1])
            self.z.clamp_(min=xyz_low[2] + eps, max=xyz_high[2] - eps)

    def xyz_min(self) -> Tensor:
        r"""
        Returns the front lower left corner position of the hodoscope in meters.
        """
        return torch.tensor([self.xy[0].item()-self.xyz_span[0].item() / 2, 
                             self.xy[1].item()-self.xyz_span[1].item() / 2, 
                             self.z.item()-self.xyz_span[2].item()])

    def xyz_max(self) -> Tensor:
        r"""
        Returns the back upper right corner position of the hodoscope in meters.
        """
        return torch.tensor([self.xy[0].item() + self.xyz_span[0].item() / 2, 
                             self.xy[1].item() + self.xyz_span[1].item() / 2, 
                             self.z.item()])