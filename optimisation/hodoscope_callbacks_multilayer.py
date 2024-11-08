from typing import List
import seaborn as sns
from matplotlib.gridspec import GridSpec

import matplotlib.lines as mlines
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

import torch
from tomopt.optimisation.callbacks import Callback, MetricLogger

from volume.hodoscopelayer import HodoscopeDetectorLayer
from volume.panel import SigmoidHodoscopeDetectorPanel


class HodoscopeMetricLogger(MetricLogger):
    r"""
    Logger for use with :class:`~volume.hodoscopelayer.HodoscopeDetectorLayer`, which is an 
    implementation of :class:`~tomopt.optimisation.monitors.MetricLogger`.

    Provides live feedback during training showing a variety of metrics to help highlight problems or test hyper-parameters without completing a full training.
    If `show_plots` is false, will instead print training and validation losses at the end of each epoch. For notebooks, `show_plots` is true by default.
    The full history is available as a dictionary by calling :meth:`~tomopt.optimisation.callbacks.monitors.MetricLogger.get_loss_history`.
    Additionally, a gif of the optimisation can be saved.
    
    Both hodoscopes and the panels inside are plotted in xz and yz views. For the top view (xy) only hodoscopes are plotted. 
    
    Arguments:
        gif_filename: optional savename for recording a gif of the optimisation process (None -> gif saved in callback savepath)
            The savename will be appended to the callback savepath ("train_weights" by default)
        gif_length: If saving gifs, controls the total length in seconds
        show_plots: whether to provide live plots during optimisation in notebooks
    """

    def _reset(self) -> None:
        det = self.wrapper.volume.get_detectors()[0]
        if isinstance(det, HodoscopeDetectorLayer):
            self.uses_sigmoid_panels = isinstance(det.hodoscopes[0].panels[0], SigmoidHodoscopeDetectorPanel)
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
            n_above=0
            n_below=0
            for det in self.wrapper.get_detectors():
                if (det.pos=='above'): 
                    axes = self.above_det
                    if (n_above==0): 
                        for ax in axes:
                            ax.clear()
                        l, s ,pl, ps = [], [], [], []
                    n_above+=1
               
                else: 
                    axes = self.below_det
                    if (n_below==0): 
                        for ax in axes:
                            ax.clear()
                        l, s ,pl, ps = [], [], [], []
                    n_below+=1
                
                if not isinstance(det, HodoscopeDetectorLayer):
                    raise ValueError(f"Detector {det} is not a PanelDetectorLayer")
                for h in det.hodoscopes:
                    #append hodoscope positions, sizes
                    l.append(np.concatenate((h.xy.detach().cpu().numpy(), h.z.detach().cpu().numpy())))
                    s.append(h.xyz_span.detach().cpu().numpy())
                    #append panel positions, sizes
                    for p in h.panels:
                        pl.append(np.concatenate((p.xy.detach().cpu().numpy(), p.z.detach().cpu().numpy())))
                        ps.append(p.xy_span.detach().cpu().numpy())
                loc, span ,p_loc, p_span= np.array(l), np.array(s), np.array(pl), np.array(ps)

                

                lw = self.wrapper.volume.lw.detach().cpu().numpy()
                axes[2].add_patch(patches.Rectangle((0, 0), lw[0], lw[1], linewidth=1, edgecolor="black", facecolor="none", hatch="x"))  # volume

                lw_mu = [self.wrapper.mu_generator.x_range, self.wrapper.mu_generator.y_range]
                axes[2].add_patch(patches.Rectangle((lw_mu[0][0],lw_mu[1][0]), lw_mu[0][1] - lw_mu[0][0], lw_mu[1][1] - lw_mu[1][0], linewidth=1, edgecolor="green", facecolor="none", hatch="x"))  # volume

                
                for h in range(len(loc)):
                    # plot panels
                    for p in range(len(p_loc)):
                        axes[0].add_line(
                            mlines.Line2D((p_loc[p, 0] - (p_span[p, 0] / 2), p_loc[p, 0] + (p_span[p, 0] / 2)), (p_loc[p, 2], p_loc[p, 2]), linewidth=2, color='red')
                        ) # xz
                        axes[1].add_line(
                            mlines.Line2D((p_loc[p, 1] - (p_span[p, 1] / 2), p_loc[p, 1] + (p_span[p, 1] / 2)), (p_loc[p, 2], p_loc[p, 2]), linewidth=2, color='red')
                        )# yz

                    #plot hodoscopes    
                    axes[0].add_patch(
                        patches.Rectangle(
                            (loc[h, 0] - (span[h, 0] / 2), loc[h, 2]),
                            span[h, 0],
                            -span[h, 2],
                            linewidth=1,
                            edgecolor=palette[h],
                            facecolor="none",
                        )
                    ) # xz
                    
                    axes[1].add_patch(
                        patches.Rectangle(
                            (loc[h, 1] - (span[h, 1] / 2), loc[h, 2]),
                            span[h, 1],
                            -span[h, 2],
                            linewidth=1,
                            edgecolor=palette[h],
                            facecolor="none",
                        )
                    ) # yz

                    axes[2].add_patch(
                        patches.Rectangle(
                            (loc[h, 0] - (span[h, 0] / 2), loc[h, 1] - (span[h, 1] / 2)),
                            span[h, 0],
                            span[h, 1],
                            linewidth=1,
                            edgecolor=palette[h],
                            facecolor="none",
                        )
                    )  # xy 

                if self.uses_sigmoid_panels: #TODO: make necessary modifications for the sigmoid hodoscope
                    self.panel_smoothness.clear()
                    with torch.no_grad():
                        panel= det.hodoscopes[0].panels[0]
                        width = panel.get_scaled_xy_span()[0].cpu().item()
                        centre = panel.xy[0].cpu().item()
                        x = torch.linspace(-width, width, 50)[:, None]
                        y = panel.sig_model(x + centre)[:, 0]
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

        for ax, dets in zip((self.above_det, self.below_det), (self.get_detectors_above(), self.get_detectors_below())):
            if not isinstance(dets[0], HodoscopeDetectorLayer):
                raise ValueError(f"Detector {det[0]} is not a HodoscopeDetectorLayer")
            lw, z = [max([det.lw[0].detach().cpu() for det in dets]), max([det.lw[1].detach().cpu() for det in dets])], max([det.z.detach().cpu() for det in dets])
            z_min = min([det.z.detach().cpu() for det in dets])
            sizes = torch.stack([h.xyz_span.detach().cpu() for det in dets for h in det.hodoscopes ], dim=0)[:,:2] / 2
            poss = torch.stack([h.xy.detach().cpu() for det in dets for h in det.hodoscopes], dim=0)
            xy_min, xy_max = (poss - sizes).min(0).values, (poss + sizes).max(0).values
            margin = max(lw) / 2

            ax[0].set_xlim(min([1, xy_min[0].item()]) - (lw[0] / 2), max([lw[0].item(), xy_max[0].item()]) + (lw[0] / 2))
            ax[1].set_xlim(min([1, xy_min[1].item()]) - (lw[1] / 2), max([lw[1].item(), xy_max[1].item()]) + (lw[1] / 2))
            ax[2].set_xlim(xy_min.min() - margin, xy_max.max() + margin)
            ax[0].set_ylim(z_min - (1.25 * dets[0].size), z + (0.25 * dets[0].size))
            ax[1].set_ylim(z_min - (1.25 * dets[0].size), z + (0.25 * dets[0].size))
            ax[2].set_ylim(xy_min.min() - margin, xy_max.max() + margin)
            ax[2].set_aspect("equal", "box")

        if self.uses_sigmoid_panels:
            self.panel_smoothness.set_xlim((-2, 2))
            self.panel_smoothness.set_xlabel("Panel model (arb. pos.)", fontsize=0.8 * self.lbl_sz, color=self.lbl_col)

    def get_detectors_above(self) -> List[HodoscopeDetectorLayer]:
        dets = []
        for det in self.wrapper.volume.get_detectors():
            if det.pos=='above': dets.append(det)
        return dets
    
    def get_detectors_below(self)-> List[HodoscopeDetectorLayer]:
        dets = []
        for det in self.wrapper.volume.get_detectors():
            if det.pos=='below': dets.append(det)
        return dets



class NoMoreNaNs(Callback):
    r"""
    Prior to parameter updates, this callback will check and set any NaN gradients to zero.
    Updates based on NaN gradients will set the parameter value to NaN.
    As for this benchmark the only learnable parameters are the hodoscope xyz positions, this callback checks the gradients of 
    these parameters.

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
            else:
                raise NotImplementedError(f"NoMoreNaNs does not yet support {type(l)}")
