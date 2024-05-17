from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import imageio
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from fastprogress.fastprogress import IN_NOTEBOOK
from matplotlib.gridspec import GridSpec

if IN_NOTEBOOK:
    from IPython.display import display

import torch
from torch import Tensor

from tomopt.volume import PanelDetectorLayer, SigmoidDetectorPanel
from tomopt.optimisation.callbacks import Callback, EvalMetric, MetricLogger, PostWarmupCallback
from volume.hodoscopelayer import HodoscopeDetectorLayer
from volume.panel import HodoscopeDetectorPanel, SigmoidHodoscopeDetectorPanel


r"""
Provides callbacks for affecting optimisation gradients
"""

__all__ = ["HodoscopeMetricLogger", "HodNoMoreNaNs", "PanelUpdateLimiter", 
           "SigmoidPanelSmoothnessSchedule", "PanelCentring"]


class MetricLogger(Callback):
    r"""
    Provides live feedback during training showing a variety of metrics to help highlight problems or test hyper-parameters without completing a full training.
    If `show_plots` is false, will instead print training and validation losses at the end of each epoch.
    The full history is available as a dictionary by calling :meth:`~tomopt.optimisation.callbacks.monitors.MetricLogger.get_loss_history`.
    Additionally, a gif of the optimisation can be saved.

    Arguments:
        gif_filename: optional savename for recording a gif of the optimisation process (None -> no gif)
            The savename will be appended to the callback savepath
        gif_length: If saving gifs, controls the total length in seconds
        show_plots: whether to provide live plots during optimisation in notebooks
    """

    tk_sz = 16
    tk_col = "black"
    lbl_sz = 24
    lbl_col = "black"
    leg_sz = 16
    cat_palette = "tab10"
    style = {"style": "whitegrid", "rc": {"patch.edgecolor": "none"}}
    h_mid = 8
    w_mid = h_mid * 16 / 9

    def __init__(self, gif_filename: Optional[str] = "optimisation_history.gif", gif_length: float = 10.0, show_plots: bool = IN_NOTEBOOK):
        self.gif_filename, self.gif_length, self.show_plots = gif_filename, gif_length, show_plots

    def _reset(self) -> None:
        r"""
        Resets plots and logs for a new optimisation
        """

        self.loss_vals: Dict[str, List[float]] = {"Training": [], "Validation": []}
        self.best_loss: float = math.inf
        self.val_epoch_results: Optional[Tuple[float, Optional[float]]] = None
        self.metric_cbs: List[EvalMetric] = []
        self.n_trn_batches = len(self.wrapper.fit_params.trn_passives) // self.wrapper.fit_params.passive_bs
        self._buffer_files: List[str] = []

        self.metric_vals: List[List[float]] = [[] for _ in self.wrapper.fit_params.metric_cbs]
        self.main_metric_idx: Optional[int] = None
        self.lock_to_metric: bool = False
        if len(self.wrapper.fit_params.metric_cbs) > 0:
            self.main_metric_idx = 0
            for i, c in enumerate(self.wrapper.fit_params.metric_cbs):
                if c.main_metric:
                    self.main_metric_idx = i
                    self.lock_to_metric = True
                    break
        self._prep_plots()
        if self.show_plots:
            self.display = display(self.fig, display_id=True)

    def on_train_begin(self) -> None:
        r"""
        Prepare for new training
        """

        super().on_train_begin()
        self._reset()

    def on_epoch_begin(self) -> None:
        r"""
        Prepare to track new loss and snapshot the plots if training
        """

        self.tmp_loss, self.batch_cnt, self.volume_cnt = 0.0, 0, 0
        if self.gif_filename is not None and self.wrapper.fit_params.state == "train" and self.show_plots:
            self._snapshot_monitor()

    def on_backwards_end(self) -> None:
        r"""
        Records the training loss for the latest volume batch
        """

        if self.wrapper.fit_params.state == "train":
            self.loss_vals["Training"].append(self.wrapper.fit_params.mean_loss.data.item() if self.wrapper.fit_params.mean_loss is not None else math.inf)

    def on_volume_batch_end(self) -> None:
        r"""
        Grabs the validation losses for the latest volume batch
        """

        if self.wrapper.fit_params.state == "valid":
            self.tmp_loss += self.wrapper.fit_params.mean_loss.data.item() if self.wrapper.fit_params.mean_loss is not None else math.inf
            self.batch_cnt += 1

    def on_epoch_end(self) -> None:
        r"""
        If validation epoch finished, record validation losses, compute info and update plots
        """

        if self.wrapper.fit_params.state == "valid":
            self.loss_vals["Validation"].append(self.tmp_loss / self.batch_cnt)

            for i, c in enumerate(self.wrapper.fit_params.metric_cbs):
                self.metric_vals[i].append(c.get_metric())
            if self.loss_vals["Validation"][-1] <= self.best_loss:
                self.best_loss = self.loss_vals["Validation"][-1]

            if self.show_plots:
                self.update_plot()
                self.display.update(self.fig)
            else:
                self.print_losses()

            m = None
            if self.lock_to_metric:
                m = self.metric_vals[self.main_metric_idx][-1]
                if not self.wrapper.fit_params.metric_cbs[self.main_metric_idx].lower_metric_better:
                    m *= -1
            self.val_epoch_results = self.loss_vals["Validation"][-1], m

    def print_losses(self) -> None:
        r"""
        Print training and validation losses for the last epoch
        """

        p = f'Epoch {len(self.loss_vals["Validation"])}: '
        p += f'Training = {np.mean(self.loss_vals["Training"][-self.n_trn_batches:]):.2E} | '
        p += f'Validation = {self.loss_vals["Validation"][-1]:.2E}'
        for m, v in zip(self.wrapper.fit_params.metric_cbs, self.metric_vals):
            p += f" {m.name} = {v[-1]:.2E}"
        print(p)

    def update_plot(self) -> None:
        r"""
        Updates the plot(s).
        """

        # Loss
        self.loss_ax.clear()
        with sns.axes_style(**self.style), sns.color_palette(self.cat_palette):
            self.loss_ax.plot(
                (1 / self.n_trn_batches)
                + np.linspace(0, len(self.loss_vals["Validation"]), self.n_trn_batches * len(self.loss_vals["Validation"]), endpoint=False),
                self.loss_vals["Training"],
                label="Training",
            )
            x = range(1, len(self.loss_vals["Validation"]) + 1)
            self.loss_ax.plot(x, self.loss_vals["Validation"], label="Validation")
            self.loss_ax.plot([1 / self.n_trn_batches, x[-1]], [self.best_loss, self.best_loss], label=f"Best = {self.best_loss:.3E}", linestyle="--")
            self.loss_ax.legend(loc="upper right", fontsize=0.8 * self.leg_sz)
            self.loss_ax.grid(True, which="both")
            self.loss_ax.set_xlim(1 / self.n_trn_batches, x[-1])
            self.loss_ax.set_ylabel("Loss", fontsize=0.8 * self.lbl_sz, color=self.lbl_col)

        if len(self.loss_vals["Validation"]) > 1:
            # Metrics
            if self.main_metric_idx is not None:
                self.metric_ax.clear()
                with sns.axes_style(**self.style), sns.color_palette(self.cat_palette) as palette:
                    x = range(self.n_trn_batches, self.n_trn_batches * len(self.loss_vals["Validation"]) + 1, self.n_trn_batches)
                    y = self.metric_vals[self.main_metric_idx]
                    self.metric_ax.plot(x, y, color=palette[1])
                    best = np.nanmin(y) if self.wrapper.fit_params.metric_cbs[self.main_metric_idx].lower_metric_better else np.nanmax(y)
                    self.metric_ax.plot([1, x[-1]], [best, best], label=f"Best = {best:.3E}", linestyle="--", color=palette[2])
                    self.metric_ax.legend(loc="upper left", fontsize=0.8 * self.leg_sz)
                    self.metric_ax.grid(True, which="both")
                    self.metric_ax.set_xlim(1 / self.n_trn_batches, x[-1])
                    self.metric_ax.set_xlabel("Epoch", fontsize=0.8 * self.lbl_sz, color=self.lbl_col)
                    self.metric_ax.set_ylabel(self.wrapper.fit_params.metric_cbs[self.main_metric_idx].name, fontsize=0.8 * self.lbl_sz, color=self.lbl_col)

    def on_train_end(self) -> None:
        r"""
        Cleans up plots, and optionally creates a gif of the training history
        """

        if self.gif_filename is not None and self.show_plots:
            self._snapshot_monitor()
            self._create_gif()
        plt.clf()  # prevent plot be shown twice
        self.metric_cbs = self.wrapper.fit_params.metric_cbs  # Copy reference since fit_params gets set to None at end of training

    def get_loss_history(self) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        r"""
        Get the current history of losses and metrics

        Returns:
            history: tuple of ordered dictionaries: first with losses, second with validation metrics
        """

        history: Tuple[Dict[str, List[float]], Dict[str, List[float]]] = ({}, {})
        history[0]["Training"] = self.loss_vals["Training"]
        history[0]["Validation"] = self.loss_vals["Validation"]
        for v, c in zip(self.metric_vals, self.metric_cbs):
            history[1][c.name] = v
        return history

    def get_results(self, loaded_best: bool) -> Dict[str, float]:
        idx: int
        if loaded_best:
            if self.lock_to_metric:
                idx = int(
                    np.nanargmin(self.metric_vals[self.main_metric_idx])
                    if self.metric_cbs[self.main_metric_idx].lower_metric_better
                    else np.nanargmax(self.metric_vals[self.main_metric_idx])
                )
            else:
                idx = int(np.nanargmin(self.loss_vals["Validation"]))
        else:
            idx = -1

        results = {}
        results["loss"] = self.loss_vals["Validation"][idx]
        if len(self.metric_cbs) > 0:
            for c, v in zip(self.metric_cbs, np.array(self.metric_vals)[:, idx]):
                results[c.name] = v
        return results

    def _snapshot_monitor(self) -> None:
        r"""
        Saves an image of all the plots in their current state
        """

        self._buffer_files.append(self.wrapper.fit_params.cb_savepath / f"temp_monitor_{len(self._buffer_files)}.png")
        self.fig.savefig(self._buffer_files[-1], bbox_inches="tight")

    def _build_grid_spec(self) -> GridSpec:
        r"""
        Returns:
            The layout object for the plots
        """

        return self.fig.add_gridspec(3 + (self.main_metric_idx is None), 1)

    def _prep_plots(self) -> None:
        r"""
        Creates the plots for a new optimisation
        """

        if self.show_plots:
            with sns.axes_style(**self.style):
                self.fig = plt.figure(figsize=(self.w_mid, self.w_mid), constrained_layout=True)
                self.grid_spec = self._build_grid_spec()
                self.loss_ax = self.fig.add_subplot(self.grid_spec[:3, :])
                if self.main_metric_idx is not None:
                    self.metric_ax = self.fig.add_subplot(self.grid_spec[4:5, :])
                self.loss_ax.tick_params(axis="x", labelsize=0.8 * self.tk_sz, labelcolor=self.tk_col)
                self.loss_ax.tick_params(axis="y", labelsize=0.8 * self.tk_sz, labelcolor=self.tk_col)
                self.loss_ax.set_ylabel("Loss", fontsize=0.8 * self.lbl_sz, color=self.lbl_col)
                if self.main_metric_idx is not None:
                    self.metric_ax.tick_params(axis="x", labelsize=0.8 * self.tk_sz, labelcolor=self.tk_col)
                    self.metric_ax.tick_params(axis="y", labelsize=0.8 * self.tk_sz, labelcolor=self.tk_col)
                    self.metric_ax.set_xlabel("Epoch", fontsize=0.8 * self.lbl_sz, color=self.lbl_col)
                    self.metric_ax.set_ylabel(self.wrapper.fit_params.metric_cbs[self.main_metric_idx].name, fontsize=0.8 * self.lbl_sz, color=self.lbl_col)

    def _create_gif(self) -> None:
        r"""
        Combines plot snapshots into a gif
        """

        with imageio.get_writer(
            self.wrapper.fit_params.cb_savepath / self.gif_filename, mode="I", duration=self.gif_length / len(self._buffer_files)
        ) as writer:
            for filename in self._buffer_files:
                image = imageio.imread(filename)
                writer.append_data(image)


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
            for axes, det in zip([self.above_det, self.below_det], self.wrapper.get_detectors()):
                l, s ,pl, ps = [], [], [], []
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

                for ax in axes:
                    ax.clear()

                lw = self.wrapper.volume.lw.detach().cpu().numpy()
                axes[2].add_patch(patches.Rectangle((0, 0), lw[0], lw[1], linewidth=1, edgecolor="black", facecolor="none", hatch="x"))  # volume

                
                for h in range(len(loc)):
                    # plot panels
                    for p in range(len(p_loc)):
                        axes[0].add_line(
                            mlines.Line2D((p_loc[p, 0] - (p_span[p, 0] / 2), p_loc[p, 0] + (p_span[p, 0] / 2)), (p_loc[p, 2], p_loc[p, 2]), linewidth=2, color=palette[p], alpha=0.6)
                        ) # xz
                        axes[1].add_line(
                            mlines.Line2D((p_loc[p, 1] - (p_span[p, 1] / 2), p_loc[p, 1] + (p_span[p, 1] / 2)), (p_loc[p, 2], p_loc[p, 2]), linewidth=2, color=palette[p], alpha=0.6)
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
                            (loc[h, 1] - (span[h, 0] / 2), loc[h, 2]),
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

                for p in range(len(p_loc)):
                    # plot muon hits
                    self.above_det[0].plot(self.wrapper.fit_params.mu.get_hits()['above']['reco_xyz'][:,p,0].detach().cpu().numpy(), 
                            self.wrapper.fit_params.mu.get_hits()['above']['reco_xyz'][:,p,-1].detach().cpu().numpy(),
                            linewidth=0, marker = '.', color = palette[p], alpha = 0.1)
                    self.below_det[0].plot(self.wrapper.fit_params.mu.get_hits()['below']['reco_xyz'][:,p,0].detach().cpu().numpy(), 
                            self.wrapper.fit_params.mu.get_hits()['below']['reco_xyz'][:,p,-1].detach().cpu().numpy(),
                            linewidth=0, marker = '.', color = palette[p], alpha = 0.1)
                    self.above_det[1].plot(self.wrapper.fit_params.mu.get_hits()['above']['reco_xyz'][:,p,1].detach().cpu().numpy(), 
                            self.wrapper.fit_params.mu.get_hits()['above']['reco_xyz'][:,p,-1].detach().cpu().numpy(),
                            linewidth=0, marker = '.', color = palette[p], alpha = 0.1)
                    self.below_det[1].plot(self.wrapper.fit_params.mu.get_hits()['below']['reco_xyz'][:,p,1].detach().cpu().numpy(), 
                            self.wrapper.fit_params.mu.get_hits()['below']['reco_xyz'][:,p,-1].detach().cpu().numpy(),
                            linewidth=0, marker = '.', color = palette[p], alpha = 0.1)


                # muon generation surface
                d = np.tan(np.pi/12) * (self.wrapper.volume.h.detach().cpu().item() - self.wrapper.volume.get_passive_z_range()[0].detach().cpu().item() + self.wrapper.volume.passive_size)
                self.above_det[0].add_line(mlines.Line2D((-d, lw[0]+d), (self.wrapper.volume.h, self.wrapper.volume.h), linewidth=2, color="black", linestyle="dotted"))
                self.above_det[1].add_line(mlines.Line2D((-d, lw[1]+d), (self.wrapper.volume.h, self.wrapper.volume.h), linewidth=2, color="black", linestyle="dotted"))
                axes[2].add_patch(patches.Rectangle((-d, -d), lw[0]+2*d, lw[1]+2*d, linewidth=1, edgecolor="grey", hatch=".", facecolor="none", alpha=0.5))
                    
                if self.uses_sigmoid_panels:
                    self.panel_smoothness.clear()
                    with torch.no_grad():
                        hodoscope= det.hodoscopes[0]
                        width = hodoscope.xyz_span[0].cpu().item()
                        centre = hodoscope.xy[0].cpu().item()
                        x = torch.linspace(-width, width, 50)[:, None]
                        y = hodoscope.panels[0].sig_model(x + centre)[:, 0]
                        self.panel_smoothness.plot(2 * x.cpu().numpy() / width, y.cpu().numpy())

            self._set_axes_labels()

    def _build_grid_spec(self) -> GridSpec:
        r"""
        Returns:
            The layout object for the plots
        """

        self.n_dets = len(self.wrapper.get_detectors())
        return self.fig.add_gridspec(4 + (self.main_metric_idx is None), 3 + self.uses_sigmoid_panels)

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
            lw, z = self.wrapper.volume.lw.detach().cpu(), det.z.detach().cpu()
            sizes = torch.stack([h.xyz_span.detach().cpu() for h in det.hodoscopes], dim=0)[:,:2] / 2
            poss = torch.stack([h.xy.detach().cpu() for h in det.hodoscopes], dim=0)
            xy_min, xy_max = (poss - sizes).min(0).values, (poss + sizes).max(0).values
            margin = lw.max() / 2

            ax[0].set_xlim(min([1, xy_min[0].item()]) - (lw[0] / 2), max([lw[0].item(), xy_max[0].item()]) + (lw[0] / 2))
            ax[1].set_xlim(min([1, xy_min[1].item()]) - (lw[1] / 2), max([lw[1].item(), xy_max[1].item()]) + (lw[1] / 2))
            ax[2].set_xlim(-0.75*lw[0] - margin, 1.75*lw[0] + margin)
            ax[0].set_ylim(z - (1.25 * det.size), z + (0.25 * det.size))
            ax[1].set_ylim(z - (1.25 * det.size), z + (0.25 * det.size))
            ax[2].set_ylim(-0.75*lw[1] - margin, 1.75*lw[1] + margin)
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
            if isinstance(l, PanelDetectorLayer):
                for p in l.panels:
                    if l.type_label == "heatmap":
                        torch.nan_to_num_(p.mu.grad, 0)
                        torch.nan_to_num_(p.norm.grad, 0)
                        torch.nan_to_num_(p.sig.grad, 0)
                        torch.nan_to_num_(p.z.grad, 0)
                    else:
                        torch.nan_to_num_(p.xy.grad, 0)
                        torch.nan_to_num_(p.z.grad, 0)
                        torch.nan_to_num_(p.xy_span.grad, 0)
            elif isinstance(l, HodoscopeDetectorLayer):
                for h in l.hodoscopes:
                    torch.nan_to_num_(h.xy.grad, 0)
                    torch.nan_to_num_(h.z.grad, 0)   
            else:
                raise NotImplementedError(f"NoMoreNaNs does not yet support {type(l)}")


class SigmoidPanelSmoothnessSchedule(PostWarmupCallback):
    r"""
    Creates an annealing schedule for the smooth attribute of :class:`~tomopt.volume.panel.SigmoidDetectorPanel`.
    This can be used to move from smooth, unphysical panel with high sensitivity outside the physical panel boundaries,
    to one with sharper decrease in resolution | efficiency at the edge, and so more closely resembles a physical panel, whilst still being differentiable.

    Arguments:
        smooth_range: tuple of initial and final values for the smooth attributes of all panels in the volume.
            A base-10 log schedule used over the number of epochs-total number of warmup epochs.
    """

    def __init__(self, smooth_range: Tuple[float, float]):
        self.smooth_range = smooth_range

    def _activate(self) -> None:
        r"""
        When the schedule begins, computes the appropriate smooth value at each up-coming epoch.
        """

        super()._activate()
        self.offset = self.wrapper.fit_params.epoch - 1
        self.smooth = torch.logspace(np.log10(self.smooth_range[0]), np.log10(self.smooth_range[1]), self.wrapper.fit_params.n_epochs - self.offset)

    def on_train_begin(self) -> None:
        r"""
        Sets all :class:`~tomopt.volume.panel.SigmoidDetectorPanel` s to their initial smooth values.
        """

        super().on_train_begin()
        self._set_smooth(Tensor([self.smooth_range[0]]))

    def _set_smooth(self, smooth: Tensor) -> None:
        r"""
        Sets the smooth values for all :class:`~tomopt.volume.panel.SigmoidDetectorPanel  in the detector.

        Arguments:
            smooth: smooth values for every :class:`~tomopt.volume.panel.SigmoidDetectorPanel` in the volume.
        """

        for det in self.wrapper.volume.get_detectors():
            if isinstance(det, PanelDetectorLayer):
                for p in det.panels:
                    if isinstance(p, SigmoidDetectorPanel):
                        p.smooth = smooth
            elif isinstance(det, HodoscopeDetectorLayer):
                for h in det.hodoscopes:
                    for p in h.panels:
                        if isinstance(p, SigmoidHodoscopeDetectorPanel):
                            p.smooth = smooth

    def on_epoch_begin(self) -> None:
        r"""
        At the start of each training epoch, will anneal the :class:`~tomopt.volume.panel.SigmoidDetectorPanel` s' smooth attributes, if the callback is active.
        """

        super().on_epoch_begin()
        if self.active:
            if self.wrapper.fit_params.state == "train":
                self._set_smooth(self.smooth[self.wrapper.fit_params.epoch - self.offset - 1])


class PanelUpdateLimiter(Callback):
    r"""
    Limits the maximum difference that optimisers can make to panel parameters, to prevent them from being affected by large updates from anomolous gradients.
    This is enacted by a hard-clamping based on the initial and final parameter values before/after each update step.

    Arguments:
        max_xy_step: maximum update in xy position of panels
        max_z_step: maximum update in z position of panels
        max_xy_span_step: maximum update in xy_span position of panels
    """

    def __init__(
        self, max_xy_step: Optional[Tuple[float, float]] = None, max_z_step: Optional[float] = None, max_xy_span_step: Optional[Tuple[float, float]] = None
    ):
        self.max_xy_step = Tensor(max_xy_step) if max_xy_step is not None else None
        self.max_z_step = Tensor([max_z_step]) if max_z_step is not None else None
        self.max_xy_span_step = Tensor(max_xy_span_step) if max_xy_span_step is not None else None

    def on_backwards_end(self) -> None:
        r"""
        Records the current paramaters of each panel before they are updated.
        """

        self.panel_params: List[Dict[str, Tensor]] = []
        for det in self.wrapper.volume.get_detectors():
            if isinstance(det, PanelDetectorLayer):
                for panel in det.panels:
                    self.panel_params.append({"xy": panel.xy.detach().clone(), "z": panel.z.detach().clone(), "xy_span": panel.xy_span.detach().clone()})
            elif isinstance(det, HodoscopeDetectorLayer):
                for h in det.hodoscopes:
                    for panel in h.panels:
                        self.panel_params.append({"xy": panel.xy.detach().clone(), "z": panel.z.detach().clone(), "xy_span": panel.xy_span.detach().clone()})
    
    def on_step_end(self) -> None:
        r"""
        After the update step, goes through and hard-clamps parameter updates based on the difference between their current values
        and values before the update step.
        """

        with torch.no_grad():
            panel_idx = 0
            for det in self.wrapper.volume.get_detectors():
                if isinstance(det, PanelDetectorLayer):
                    for panel in det.panels:
                        if self.max_xy_step is not None:
                            delta = panel.xy - self.panel_params[panel_idx]["xy"]
                            panel.xy.data = torch.where(
                                delta.abs() > self.max_xy_step, self.panel_params[panel_idx]["xy"] + (torch.sign(delta) * self.max_xy_step), panel.xy
                            )

                        if self.max_z_step is not None:
                            delta = panel.z - self.panel_params[panel_idx]["z"]
                            panel.z.data = torch.where(
                                delta.abs() > self.max_z_step, self.panel_params[panel_idx]["z"] + (torch.sign(delta) * self.max_z_step), panel.z
                            )

                        if self.max_xy_span_step is not None:
                            delta = panel.xy_span - self.panel_params[panel_idx]["xy_span"]
                            panel.xy_span.data = torch.where(
                                delta.abs() > self.max_xy_span_step,
                                self.panel_params[panel_idx]["xy_span"] + (torch.sign(delta) * self.max_xy_span_step),
                                panel.xy_span,
                            )
                        panel_idx += 1

                elif isinstance(det, HodoscopeDetectorLayer):
                    for h in det.hodoscopes:
                        for panel in h.panels:
                            if self.max_xy_step is not None:
                                delta = panel.xy - self.panel_params[panel_idx]["xy"]
                                panel.xy.data = torch.where(
                                    delta.abs() > self.max_xy_step, self.panel_params[panel_idx]["xy"] + (torch.sign(delta) * self.max_xy_step), panel.xy
                                )

                            if self.max_z_step is not None:
                                delta = panel.z - self.panel_params[panel_idx]["z"]
                                panel.z.data = torch.where(
                                    delta.abs() > self.max_z_step, self.panel_params[panel_idx]["z"] + (torch.sign(delta) * self.max_z_step), panel.z
                                )

                            if self.max_xy_span_step is not None:
                                delta = panel.xy_span - self.panel_params[panel_idx]["xy_span"]
                                panel.xy_span.data = torch.where(
                                    delta.abs() > self.max_xy_span_step,
                                    self.panel_params[panel_idx]["xy_span"] + (torch.sign(delta) * self.max_xy_span_step),
                                    panel.xy_span,
                                )
                            panel_idx += 1


class PanelCentring(Callback):
    """
    Callback class for panel centring in the optimisation process.

    This callback is used to centre the panels of PanelDetectorLayer objects
    by setting their xy coordinates to the mean xy value of all panels in the layer.

    This update takes place after the panel positions have been updated in the optimisation process.
    """

    def on_step_end(self) -> None:
        """
        Updates the xy coordinates of all panels in the PanelDetectorLayer objects after they have be updated, based on their current mean xy position.
        """
        for l in self.wrapper.volume.get_detectors():
            if isinstance(l, PanelDetectorLayer):
                xy = []
                for p in l.panels:
                    xy.append(p.xy.detach().cpu().numpy())
                mean_xy = Tensor(np.mean(xy, 0), device=self.wrapper.device)
                for p in l.panels:
                    p.xy.data = mean_xy
            elif isinstance(l, HodoscopeDetectorLayer):
                xy = []
                for h in l.hodoscopes:
                    for p in h.panels:
                        xy.append(p.xy.detach().cpu().numpy())
                mean_xy = Tensor(np.mean(xy, 0), device=self.wrapper.device)
                for h in l.hodoscopes:
                    for p in h.panels:
                        p.xy.data = mean_xy
            else:
                raise NotImplementedError(f"PanelCentring does not yet support {type(l)}")
