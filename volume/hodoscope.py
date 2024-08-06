from typing import Tuple, List, Optional, Union
from tomopt.core import DEVICE
import torch
from torch import Tensor
from torch import nn
from tomopt.volume.panel import DetectorPanel, SigmoidDetectorPanel


class Hodoscope(nn.Module):
    def __init__(
        self,
        *,
        init_xyz: Tuple[float, float, float],
        init_xyz_span: Tuple[float, float, float],
        xyz_gap: Tuple[float, float, float],
        n_panels: int = 3,
        res: float = 1000,
        eff: float = 0.9,
        m2_cost: float = 1.0,
        budget: Optional[Tensor] = None,
        realistic_validation: bool = True,
        panel_type: str = "DetectorPanel",
        smooth: Union[float, Tensor] = None,
        device: torch.device = DEVICE,
    ):
        super().__init__()
        self.realistic_validation, self.device = realistic_validation, device
        # self.register_buffer("m2_cost", torch.tensor(float(m2_cost), device=self.device))
        self.xy = nn.Parameter(torch.tensor(init_xyz[:2], device=self.device))
        self.z = nn.Parameter(torch.tensor(init_xyz[-1], device=self.device))
        self.xyz_span = nn.Parameter(torch.tensor(init_xyz_span, device=self.device))
        self.xyz_gap = xyz_gap
        self.n_panels = n_panels
        self.res = res
        self.eff = eff
        self.panel_type = panel_type
        self.smooth = smooth
        self.panels = self.generate_init_panels()
        self.device = device

    def __getitem__(self, idx: int) -> DetectorPanel:
        return self.panels[idx]

    def generate_init_panels(
        self,
    ) -> Union[List[DetectorPanel], List[SigmoidDetectorPanel]]:
        r"""
        Generates Detector panels based on the xy and z position (xy, z), the span of the hodoscope (xyz_span),
        and the gap between the edge of the hodoscope and the panels (xyz_gap).

        Returns:
            DetectorPanels as a nn.ModuleList.
        """

        if self.panel_type == "DetectorPanel":
            return nn.ModuleList(
                [
                    DetectorPanel(
                        res=self.res,
                        eff=self.eff,
                        realistic_validation=self.realistic_validation,
                        init_xyz=[
                            self.xy[0],
                            self.xy[1],
                            self.z
                            - self.xyz_gap[2]
                            - (self.xyz_span[2] - 2 * self.xyz_gap[2])
                            * i
                            / (self.n_panels - 1),
                        ],
                        init_xy_span=[
                            self.xyz_span[0] - 2 * self.xyz_gap[0],
                            self.xyz_span[1] - 2 * self.xyz_gap[1],
                        ],
                        device=DEVICE,
                    )
                    for i in range(self.n_panels)
                ]
            )

        elif self.panel_type == "SigmoidDetectorPanel":
            return nn.ModuleList(
                [
                    SigmoidDetectorPanel(
                        smooth=self.smooth,
                        res=self.res,
                        eff=self.eff,
                        realistic_validation=self.realistic_validation,
                        init_xyz=[
                            self.xy[0],
                            self.xy[1],
                            self.z
                            - self.xyz_gap[2]
                            - (self.xyz_span[2] - 2 * self.xyz_gap[2])
                            * i
                            / (self.n_panels - 1),
                        ],
                        init_xy_span=[
                            self.xyz_span[0] - 2 * self.xyz_gap[0],
                            self.xyz_span[1] - 2 * self.xyz_gap[1],
                        ],
                        device=DEVICE,
                    )
                    for i in range(self.n_panels)
                ]
            )

        else:
            raise ValueError(
                f"Detector type {self.panel_type} currently not supported."
            )

    def get_xyz_min(self) -> Tuple[float, float, float]:
        return [
            self.xy[0].item() - self.xyz_span[0].item(),
            self.xy[1].item() - self.xyz_span[1].item(),
            self.z.item() - self.xyz_span[2].item(),
        ]

    def get_xyz_max(self) -> Tuple[float, float, float]:
        return [
            self.xy[0].item() + self.xyz_span[0].item(),
            self.xy[1].item() + self.xyz_span[1].item(),
            self.z.item() + self.xyz_span[2].item(),
        ]

    def get_cost(self) -> Tensor:
        return torch.sum(torch.Tensor([p.get_cost() for p in self.panels]))
