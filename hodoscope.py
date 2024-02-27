from typing import Tuple, List, Optional
from tomopt.core import DEVICE
import torch
from torch import Tensor
from torch import nn
from tomopt.volume.panel import DetectorPanel
import numpy as np

class Hodoscope(nn.Module):

    def __init__(self,
                 *,
                 init_xyz: Tuple[float, float, float],
                 init_xyz_span: Tuple[float, float, float],
                 xyz_gap: Tuple[float, float, float],
                 n_panels: int = 3,
                 res: float = 1000,
                 eff: float = 0.9,
                 m2_cost: float = 1.,
                 budget: Optional[Tensor] = None,
                 realistic_validation: bool = True,
                 device: torch.device = DEVICE):
        
        super().__init__()
        self.realistic_validation, self.device = realistic_validation, device
        # self.register_buffer("m2_cost", torch.tensor(float(m2_cost), device=self.device))
        self.xy = nn.Parameter(torch.tensor(init_xyz[:2], device=self.device))
        self.z = nn.Parameter(torch.tensor(init_xyz[2:3], device=self.device))
        self.xyz_span = nn.Parameter(torch.tensor(init_xyz_span, device=self.device))
        self.xyz_gap = torch.tensor(xyz_gap)
        self.n_panels = n_panels
        self.res = res
        self.eff = eff
        self.panels = self.generate_init_panels()

        self.device = device
    
    def __getitem__(self, idx: int) -> DetectorPanel:
        return self.panels[idx]

    def generate_init_panels(self) -> List[DetectorPanel]:

        r"""
        Generates Detector panels based on the xy and z position (xy, z), the span of the hodoscope (xyz_span), 
        and the gap between the edge of the hodoscope and the panels (xyz_gap).

        Returns:
            DetectorPanles as a nn.ModuleList.
        """
        
        return nn.ModuleList(
            [DetectorPanel(res=self.res, 
                           eff=self.eff,
                           init_xyz=[self.xy[0],
                                     self.xy[1],
                                     self.z - self.xyz_gap[2] - (self.xyz_span[2]-2*self.xyz_gap[2])*i/(self.n_panels-1)], 
                           init_xy_span=[self.xyz_span[0] - 2 * self.xyz_gap[0], self.xyz_span[1] - 2 * self.xyz_gap[1]],
                           device=DEVICE) for i in range(self.n_panels)])

    def clamp_params(self, xyz_low: Tuple[float, float, float], xyz_high: Tuple[float, float, float]) -> None:
        r"""
        Ensures that the hodoscope is centred within the supplied xyz range,
        and that the span of the hodoscope is between xyz_high/20 and xyz_high*10.
        A small random number < 1e-3 is added/subtracted to the min/max z position of the hodoscope, to ensure it doesn't overlap with other hodoscopes.

        Then, loops through panels and calls their `clamp_params` method, to ensure that panels are located within the bounds of the hodoscope.
        This is equivalent to the `conform_detector` method, but it is run internally because of the peculiarity of the `HodoscopeLayer`->`Hodoscope`->`DetectorPanel` hierarchy.

        Arguments:
            xyz_low: minimum x,y,z values for the panel centre in metres
            xyz_high: maximum x,y,z values for the panel centre in metres
        """

        with torch.no_grad():
            eps = np.random.uniform(0, 1e-3)  # prevent hits at same z due to clamping
            self.x.clamp_(min=xyz_low[0], max=xyz_high[0])
            self.y.clamp_(min=xyz_low[1], max=xyz_high[1])
            self.z.clamp_(min=xyz_low[2] + eps, max=xyz_high[2] - eps)
            self.xyz_span[0].clamp_(min=xyz_high[0] / 20, max=10 * xyz_high[0])
            self.xyz_span[1].clamp_(min=xyz_high[1] / 20, max=10 * xyz_high[1])
            self.xyz_span[2].clamp_(min=xyz_high[2] / 20, max=10 * xyz_high[2])

        xy = self.xy.detach().cpu().numpy()
        z = self.z.detach().cpu()[0]
        for p in self.panels:
            p.clamp_params(
                xyz_low=(xy[0] + self.xyz_gap[0],
                         xy[1] + self.xyz_gap[1],
                         z - self.xyz_gap[2]),
                xyz_high=(xy[0] + self.xyz_span[0] - self.xyz_gap[0],
                          xy[1] + self.xyz_span[1] - self.xyz_gap[1],
                          z + self.xyz_span[2] - self.xyz_gap[2]),
            )

    def get_cost(self) -> Tensor:
        r"""
        Returns:
            current cost of the hodoscope according to the cost of the panels it is constituted of
        """
        return torch.sum([p.get_cost() for p in self.panels])

    @property
    def x(self) -> Tensor:
        return self.xy[0]

    @property
    def y(self) -> Tensor:
        return self.xy[1]
