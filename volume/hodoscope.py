from typing import Tuple, List, Optional, Union
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
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

    def __repr__(self) -> str:
        return "Hodoscope located at z = {:.2f} m".format(self.z.data.item())

    def get_init_panels_pos(self) -> List[Tuple[float, float, float]]:
        """
        Computes the initial position of the hodoscope panels.
        Panels' xy position is the hodoscope xy position `.xy`.
        Panels' z position is initialized such that the gap between the top/bottom of the hodoscope and the first/last panel is `.xyz_gap[2]`.

        Returns:
            Panels x,y,z positions as a list of tuples.
        """
        span_z = self.xyz_span[2] - 2 * self.xyz_gap[2]
        return [
            (
                self.xy[0].item(),
                self.xy[1].item(),
                self.z.item() - self.xyz_gap[2] - span_z * i / (self.n_panels - 1),
            )
            for i in range(self.n_panels)
        ]

    def get_init_panels_span(self) -> List[Tuple[float, float]]:
        """
        Computes the initial span of the hodoscope panels.
        Panels' xy span is initialized such that the gap between the left/right edge of the hodoscope
        and the left/right edge of the panels is `.xyz_gap[0]`, `.xyz_gap[1]` along x and y respectively.

        Returns:
            Panels x,y spans as a list of tuples.
        """
        return [
            (
                self.xyz_span[0] - 2 * self.xyz_gap[0],
                self.xyz_span[1] - 2 * self.xyz_gap[1],
            )
            for _ in range(self.n_panels)
        ]

    def generate_init_panels(self) -> nn.ModuleList:
        r"""
        Generates `.n_panels` DetectorPanels or SigmoidDetectorPanels according to the chosen `.panel_type`.
        Panels' xyz position and span are computed via the `get_init_panels_pos` and `get_init_panels_span` methods.

        Returns:
            DetectorPanels instances as a nn.ModuleList.
        """

        panel_cls = (
            DetectorPanel
            if self.panel_type == "DetectorPanel"
            else SigmoidDetectorPanel
        )
        if panel_cls not in [DetectorPanel, SigmoidDetectorPanel]:
            raise ValueError(
                f"Detector type {self.panel_type} currently not supported."
            )

        panel_positions = self.get_init_panels_pos()
        panel_spans = self.get_init_panels_span()

        panels = []
        for i in range(self.n_panels):
            panel_args = {
                "res": self.res,
                "eff": self.eff,
                "realistic_validation": self.realistic_validation,
                "init_xyz": panel_positions[i],
                "init_xy_span": panel_spans[i],
                "device": self.device,
            }
            if self.panel_type == "SigmoidDetectorPanel":
                panel_args["smooth"] = self.smooth

            panels.append(panel_cls(**panel_args))

        return nn.ModuleList(panels)

    def get_xyz_min(self) -> Tuple[float, float, float]:
        r"""
        Returns:
            The position of the hodoscope bottom left corner (meters, in absolute position in the volume frame).
        """
        return (
            self.xy[0].item() - self.xyz_span[0].item() / 2,
            self.xy[1].item() - self.xyz_span[1].item() / 2,
            self.z.item() - self.xyz_span[2].item(),
        )

    def get_xyz_max(self) -> Tuple[float, float, float]:
        r"""
        Returns:
            The position of the hodoscope upper right corner (meters, in absolute position in the volume frame).
        """
        return (
            self.xy[0].item() + self.xyz_span[0].item() / 2,
            self.xy[1].item() + self.xyz_span[1].item() / 2,
            self.z.item(),
        )

    def get_cost(self) -> Tensor:
        r"""
        Returns:
            Current cost of the hodoscope, given the cost of its panels.
        """
        return torch.sum(torch.Tensor([p.get_cost() for p in self.panels]))

    def draw(self) -> None:
        """
        Draws the hodoscope and its panels.
        """
        hod_data = {
            "x": self.xy[0].detach().numpy() - self.xyz_span[0].detach().item() / 2,
            "y": self.xy[1].detach().numpy() - self.xyz_span[1].detach().item() / 2,
            "z": self.z.detach().numpy() - self.xyz_span[2].detach().item(),
            "dx": self.xyz_span[0].detach().item(),
            "dy": self.xyz_span[1].detach().item(),
            "dz": self.xyz_span[2].detach().item(),
        }

        panels_data = [
            {
                "x": p.xy[0].detach().item(),
                "y": p.xy[1].detach().item(),
                "z": p.z.detach().item(),
                "dx": p.xy_span[0].detach().item(),
                "dy": p.xy_span[1].detach().item(),
            }
            for p in self.panels
        ]

        def normalize(x, xmin, xmax):
            return (x - xmin) / (xmax - xmin)

        fig, axs = plt.subplots(ncols=3, figsize=(9, 3))
        fig.suptitle(
            f"Hodoscope at x,y,z = {self.xy[0].detach().item():.1f},{self.xy[1].detach().item():.1f},{self.z.detach().item():.1f} m",
            fontweight="bold",
            fontsize=15,
        )

        for ax in axs:
            ax.set_aspect("equal")
            ax.grid("on")

        def set_limits(ax, axis, hod_data, span_key, gap_key):
            ax.set_xlim(
                hod_data[axis] - 2 * self.xyz_gap[gap_key],
                hod_data[axis]
                + self.xyz_span[span_key].detach().item()
                + 2 * self.xyz_gap[gap_key],
            )
            ax.set_ylim(
                hod_data["z"] - 2 * self.xyz_gap[2],
                hod_data["z"] + self.xyz_span[2].detach().item() + 2 * self.xyz_gap[2],
            )

        # Plot XY view
        xz_hod = Rectangle(
            (hod_data["x"], hod_data["z"]), hod_data["dx"], hod_data["dz"]
        )
        axs[0].add_collection(
            PatchCollection([xz_hod], facecolor="green", alpha=0.2, edgecolor="green")
        )
        set_limits(axs[0], "x", hod_data, 0, 0)
        axs[0].set_xlabel(r"$x$ [m]")
        axs[0].set_ylabel(r"$z$ [m]")

        for p in panels_data:
            axs[0].axhline(
                y=p["z"],
                color="red",
                xmin=normalize(
                    p["x"] - p["dx"] / 2, axs[0].get_xlim()[0], axs[0].get_xlim()[1]
                ),
                xmax=normalize(
                    p["x"] + p["dx"] / 2, axs[0].get_xlim()[0], axs[0].get_xlim()[1]
                ),
            )

        # Plot YZ view
        yz_hod = Rectangle(
            (hod_data["y"], hod_data["z"]), hod_data["dy"], hod_data["dz"]
        )
        axs[1].add_collection(
            PatchCollection([yz_hod], facecolor="green", alpha=0.2, edgecolor="green")
        )
        set_limits(axs[1], "y", hod_data, 1, 1)
        axs[1].set_xlabel(r"$y$ [m]")
        axs[1].set_ylabel(r"$z$ [m]")

        for p in panels_data:
            axs[1].axhline(
                y=p["z"],
                color="red",
                xmin=normalize(
                    p["y"] - p["dy"] / 2, axs[1].get_xlim()[0], axs[1].get_xlim()[1]
                ),
                xmax=normalize(
                    p["y"] + p["dy"] / 2, axs[1].get_xlim()[0], axs[1].get_xlim()[1]
                ),
            )

        # Plot XZ view
        xy_hod = Rectangle(
            (hod_data["x"], hod_data["y"]), hod_data["dx"], hod_data["dy"]
        )
        xy_panels = [
            Rectangle(
                (
                    panel_data["x"] - panel_data["dx"] / 2,
                    panel_data["y"] - panel_data["dy"] / 2,
                ),
                panel_data["dx"],
                panel_data["dy"],
            )
            for panel_data in panels_data
        ]

        axs[2].add_collection(
            PatchCollection(
                [xy_hod],
                facecolor="green",
                alpha=0.2,
                edgecolor="green",
            )
        )
        axs[2].add_collection(
            PatchCollection(
                xy_panels,
                facecolor="red",
                alpha=0.2,
                edgecolor="red",
            )
        )

        axs[2].set_xlim(
            hod_data["x"] - 2 * self.xyz_gap[1],
            hod_data["x"] + self.xyz_span[0].detach().item() + 2 * self.xyz_gap[0],
        )

        axs[2].set_ylim(
            hod_data["y"] - 2 * self.xyz_gap[2],
            hod_data["y"] + self.xyz_span[1].detach().item() + 2 * self.xyz_gap[1],
        )

        axs[2].set_xlabel(r"$x$ [m]")
        axs[2].set_ylabel(r"$y$ [m]")

        # Plot hodoscope parameters
        axs[0].scatter(
            self.xy[0].detach().numpy(),
            self.z.detach().numpy(),
            color="blue",
            marker="+",
            label=r"initial $xyz$",
            s=50,
        )
        axs[1].scatter(
            self.xy[1].detach().numpy(),
            self.z.detach().numpy(),
            color="blue",
            marker="+",
            s=50,
        )
        axs[2].scatter(
            self.xy[0].detach().numpy(),
            self.xy[1].detach().numpy(),
            color="blue",
            marker="+",
            s=50,
        )

        fig.legend(loc="lower left")

        plt.tight_layout()
        plt.show()
