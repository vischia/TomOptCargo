import seaborn as sns
from matplotlib.gridspec import GridSpec

import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from fastprogress.fastprogress import IN_NOTEBOOK
from matplotlib.gridspec import GridSpec

import torch
from tomopt.optimisation.callbacks import Callback, MetricLogger

from volume.hodoscope import Hodoscope
from volume.hodoscopelayer import HodoscopeDetectorLayer
from tomopt.volume import SigmoidDetectorPanel


class HodoscopeMetricLogger(MetricLogger):
    r"""
    Logger for use with :class:`~volume.hodoscopelayer.HodoscopeDetectorLayer` s

    Arguments:
        gif_filename: optional savename for recording a gif of the optimisation process (None -> no gif)
            The savename will be appended to the callback savepath
        gif_length: If saving gifs, controls the total length in seconds
        show_plots: whether to provide live plots during optimisation in notebooks
    """

    def _reset(self) -> None:
        det = self.wrapper.volume.get_detectors()[0]
        if isinstance(det, HodoscopeDetectorLayer):
            self.uses_sigmoid_panels = isinstance(det.hodoscopes[0].panels[0], SigmoidDetectorPanel)
        else:
            self.uses_sigmoid_panels = False
        super()._reset()

    def _prep_plots(self) -> None:
        r"""
        Creates the plots for a new optimisation
        """

        super()._prep_plots()
        if self.show_plots:
            with sns.axes_style(**self.style):
                self.above_det = [self.fig.add_subplot(self.grid_spec[-2:-1, i : i + 1]) for i in range(3)]
                self.below_det = [self.fig.add_subplot(self.grid_spec[-1:, i : i + 1]) for i in range(3)]
                if self.uses_sigmoid_panels:
                    self.panel_smoothness = self.fig.add_subplot(self.grid_spec[-2:-1, -1:])
                self._set_axes_labels()

    def update_plot(self) -> None:
        r"""
        Updates the plot(s).
        """

        super().update_plot()
        with sns.axes_style(**self.style), sns.color_palette(self.cat_palette) as palette:
            for axes, det in zip([self.above_det, self.below_det], self.wrapper.get_detectors()):
                l, s = [], []
                if not isinstance(det, HodoscopeDetectorLayer):
                    raise ValueError(f"Detector {det} is not a PanelDetectorLayer")
                for p in det.hodoscopes:
                    if det.type_label == "heatmap":
                        l_val = np.concatenate((p.mu.detach().cpu().numpy().mean(axis=0), p.z.detach().cpu().numpy()))
                        s_val = p.sig.detach().cpu().numpy().mean(axis=0)
                        l.append(l_val)
                        s.append(s_val)
                    else:
                        l.append(np.concatenate((p.xy.detach().cpu().numpy(), p.z.detach().cpu().numpy())))
                        s.append(p.xyz_span.detach().cpu().numpy())
                loc, span = np.array(l), np.array(s)

                for ax in axes:
                    ax.clear()

                lw = self.wrapper.volume.lw.detach().cpu().numpy()
                axes[2].add_patch(patches.Rectangle((0, 0), lw[0], lw[1], linewidth=1, edgecolor="black", facecolor="none", hatch="x"))  # volume

                for p in range(len(loc)):
                    axes[0].add_line(
                        mlines.Line2D((loc[p, 0] - (span[p, 0] / 2), loc[p, 0] + (span[p, 0] / 2)), (loc[p, 2], loc[p, 2]), linewidth=2, color=palette[p])
                    )  # xz
                    axes[1].add_line(
                        mlines.Line2D((loc[p, 1] - (span[p, 1] / 2), loc[p, 1] + (span[p, 1] / 2)), (loc[p, 2], loc[p, 2]), linewidth=2, color=palette[p])
                    )  # yz
                    axes[2].add_patch(
                        patches.Rectangle(
                            (loc[p, 0] - (span[p, 0] / 2), loc[p, 1] - (span[p, 1] / 2)),
                            span[p, 0],
                            span[p, 1],
                            linewidth=1,
                            edgecolor=palette[p],
                            facecolor="none",
                        )
                    )  # xy

                if self.uses_sigmoid_panels:
                    self.panel_smoothness.clear()
                    with torch.no_grad():
                        hodoscope= det.hodoscopes[0]
                        #panel = det.panels[0]
                        width = hodoscope.xyz_span[0].cpu().item()
                        centre = hodoscope.xy[0].cpu().item()
                        x = torch.linspace(-width, width, 50)[:, None]
                        y = hodoscope.sig_model(x + centre)[:, 0]
                        self.panel_smoothness.plot(2 * x.cpu().numpy() / width, y.cpu().numpy())

            self._set_axes_labels()

    def _build_grid_spec(self) -> GridSpec:
        r"""
        Returns:
            The layout object for the plots
        """

        self.n_dets = len(self.wrapper.get_detectors())
        return self.fig.add_gridspec(5 + (self.main_metric_idx is None), 3 + self.uses_sigmoid_panels)

    def _set_axes_labels(self) -> None:
        r"""
        Adds styling to plots after they are cleared
        """

        for ax, x in zip(self.below_det, ["x", "y", "x"]):
            ax.set_xlabel(x, fontsize=0.8 * self.lbl_sz, color=self.lbl_col)
        for i, (ax, x) in enumerate(zip(self.above_det, ["z", "z", "y"])):
            if i == 0:
                x = "Above, " + x
            ax.set_ylabel(x, fontsize=0.8 * self.lbl_sz, color=self.lbl_col)
        for i, (ax, x) in enumerate(zip(self.below_det, ["z", "z", "y"])):
            if i == 0:
                x = "Below, " + x
            ax.set_ylabel(x, fontsize=0.8 * self.lbl_sz, color=self.lbl_col)

        for ax, det in zip((self.above_det, self.below_det), self.wrapper.get_detectors()):
            if not isinstance(det, HodoscopeDetectorLayer):
                raise ValueError(f"Detector {det} is not a HodoscopeDetectorLayer")
            lw, z = det.lw.detach().cpu(), det.z.detach().cpu()
            #sizes = torch.stack([h.xyz_span.detach().cpu() for h in det.hodoscopes], dim=0)[:,:2] / 2
            sizes = torch.stack([h.xyz_span.detach().cpu() for h in det.hodoscopes], dim=0)[:,:2] / 2
            poss = torch.stack([h.xy.detach().cpu() for h in det.hodoscopes], dim=0)
            xy_min, xy_max = (poss - sizes).min(0).values, (poss + sizes).max(0).values
            margin = lw.max() / 2

            ax[0].set_xlim(min([1, xy_min[0].item()]) - (lw[0] / 2), max([lw[0].item(), xy_max[0].item()]) + (lw[0] / 2))
            ax[1].set_xlim(min([1, xy_min[1].item()]) - (lw[1] / 2), max([lw[1].item(), xy_max[1].item()]) + (lw[1] / 2))
            ax[2].set_xlim(xy_min.min() - margin, xy_max.max() + margin)
            ax[0].set_ylim(z - (1.25 * det.size), z + (0.25 * det.size))
            ax[1].set_ylim(z - (1.25 * det.size), z + (0.25 * det.size))
            ax[2].set_ylim(xy_min.min() - margin, xy_max.max() + margin)
            ax[2].set_aspect("equal", "box")

        if self.uses_sigmoid_panels:
            self.panel_smoothness.set_xlim((-2, 2))
            self.panel_smoothness.set_xlabel("Panel model (arb. pos.)", fontsize=0.8 * self.lbl_sz, color=self.lbl_col)



class NoMoreNaNs(Callback):
    r"""
    Prior to parameter updates, this callback will check and set any NaN gradients to zero.
    Updates based on NaN gradients will set the parameter value to NaN.

    .. important::
        As new parameters are introduced, e.g. through new detector models, this callback will need to be updated.
    """

    def on_backwards_end(self) -> None:
        r"""
        Prior to optimiser updates, parameter gradients are checked for NaNs.
        """

        if hasattr(self.wrapper.volume, "budget_weights"):
            torch.nan_to_num_(self.wrapper.volume.budget_weights.grad, 0)
        for l in self.wrapper.volume.get_detectors():
            if isinstance(l, HodoscopeDetectorLayer):
                for h in l.hodoscopes:
                        torch.nan_to_num_(h.xy.grad, 0)
                        torch.nan_to_num_(h.z.grad, 0)
                        #torch.nan_to_num_(h.xyz_span.grad, 0)
            else:
                raise NotImplementedError(f"NoMoreNaNs does not yet support {type(l)}")
