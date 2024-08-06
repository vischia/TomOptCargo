from typing import Tuple, List, Optional, Union
from tomopt.core import DEVICE
import torch
from torch import Tensor
from torch import nn
from tomopt.volume.panel import DetectorPanel, SigmoidDetectorPanel


class Hodoscope(nn.Module):
    r"""
    Provides a hodoscope detection module made of multiple DetectorPanels, centered at a learnable xyz position (meters, in absolute position in the volume frame).
    While this class can be used manually, it is designed to be used by the HodoscopeDetectorLayer class.

    Instances of `DetectorPanel` or `SigmoidDetectorPanel` will be created within the hodoscope.
    Panels' xy initial position is the hodoscope xy position `.xy`.
    Panels' z position is initialized such that the gap between the top/bottom of the hodoscope and the first/last panel is `.xyz_gap[2]`.
    Panels' xy span is initialized such that the gap between the left/right edge of the hodoscope and the left/right edge of the panels is `.xyz_gap[0]`, `.xyz_gap[1]` along x and y respectively.

    The resolution and efficiency of each panel remain fixed at the specified values.

    Arguments:
        init_xyz: Initial xyz position of the top of the hodoscope (in meters in the volume frame)
        init_xyz_span: Initial xyz-span (total width) of the hodoscope (in meters in the volume frame)
        xyz_gap: Gap between the detector panels and the edges of the hodoscope (in meters)
        n_panels: The number of detection panels within the hodoscope.
        res: Resolution of the panels in m^-1, i.e., a higher value improves the precision on the hit recording
        eff: Efficiency of the hit recording of the panels, indicated as a probability [0, 1]
        m2_cost: The cost in unit currency of 1 square meter of detector
        budget: Optional required cost of the panel. Based on the span and cost per m^2, the panel will resize to meet the required cost
        realistic_validation: If True, will use the physical interpretation of the panel during evaluation
        device: Device on which to place tensors
    """
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
        """
        Generates `.n_panels` DetectorPanels or SigmoidDetectorPanels according to the chosen `.panel_type`. 
        Panels' xyz position and span are computed via the `get_init_panels_pos` and `get_init_panels_span` methods.

        Returns:
            DetectorPanels instances as a nn.ModuleList.
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
        r"""
        Returns:
            The position of the hodoscope bottom left corner (meters, in absolute position in the volume frame).
        """
        return [
            self.xy[0].item() - self.xyz_span[0].item(),
            self.xy[1].item() - self.xyz_span[1].item(),
            self.z.item() - self.xyz_span[2].item(),
        ]

    def get_xyz_max(self) -> Tuple[float, float, float]:
        r"""
        Returns:
            The position of the hodoscope upper right corner (meters, in absolute position in the volume frame).
        """
        return [
            self.xy[0].item() + self.xyz_span[0].item(),
            self.xy[1].item() + self.xyz_span[1].item(),
            self.z.item() + self.xyz_span[2].item(),
        ]

    def get_cost(self) -> Tensor:
        r"""
        Returns:
            Current cost of the hodoscope, given the cost of its panels.
        """
        return torch.sum(torch.Tensor([p.get_cost() for p in self.panels]))
