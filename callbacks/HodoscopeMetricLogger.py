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

from tomopt.volume import PanelDetectorLayer, SigmoidDetectorPanel
from tomopt.optimisation.callbacks import Callback, EvalMetric, MetricLogger
from volume.hodoscopelayer import HodoscopeDetectorLayer

class HodoscopeMetricLogger(MetricLogger):
    r"""
    Logger for use with :class:`~tomopt.volume.layer.HodoscopeDetectorLayer` s 
                     or :class:`~tomopt.volume.layer.PanelDetectorLayer` s 

    Arguments:
        gif_filename: optional savename for recording a gif of the optimisation process (None -> no gif)
            The savename will be appended to the callback savepath
        gif_length: If saving gifs, controls the total length in seconds
        show_plots: whether to provide live plots during optimisation in notebooks
    """

    def _reset(self) -> None:
        det = self.wrapper.volume.get_detectors()[0]
        if isinstance(det, PanelDetectorLayer):
            self.uses_sigmoid_panels = isinstance(det.panels[0], SigmoidDetectorPanel)
        elif isinstance(det, HodoscopeDetectorLayer):
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

                if isinstance(det, PanelDetectorLayer):
                    for p in det.panels:
                        if det.type_label == "heatmap":
                            l_val = np.concatenate((p.mu.detach().cpu().numpy().mean(axis=0), p.z.detach().cpu().numpy()))
                            s_val = p.sig.detach().cpu().numpy().mean(axis=0)
                            l.append(l_val)
                            s.append(s_val)
                        else:
                            l.append(np.concatenate((p.xy.detach().cpu().numpy(), p.z.detach().cpu().numpy())))
                            s.append(p.get_scaled_xy_span().detach().cpu().numpy())

                if isinstance(det, HodoscopeDetectorLayer):
                    for h in det.hodoscopes:
                        for p in h.panels:
                            if det.type_label == "heatmap":
                                l_val = np.concatenate((p.mu.detach().cpu().numpy().mean(axis=0), p.z.detach().cpu().numpy()))
                                s_val = p.sig.detach().cpu().numpy().mean(axis=0)
                                l.append(l_val)
                                s.append(s_val)
                            else:
                                l.append(np.concatenate((p.xy.detach().cpu().numpy(), p.z.detach().cpu().numpy())))
                                s.append(p.get_scaled_xy_span().detach().cpu().numpy())
                
                else:
                    raise ValueError(f"Detector {det} is not supported")

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
                        panel = det.panels[0]
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

        for ax, det in zip((self.above_det, self.below_det), self.wrapper.get_detectors()):
            if isinstance(det, PanelDetectorLayer):
                lw, z = det.lw.detach().cpu(), det.z.detach().cpu()
                sizes = torch.stack([p.get_scaled_xy_span().detach().cpu() for p in det.panels], dim=0) / 2
                poss = torch.stack([p.xy.detach().cpu() for p in det.panels], dim=0)
            elif isinstance(det, HodoscopeDetectorLayer):
                lw, z = det.lw.detach().cpu(), det.z.detach().cpu()
                sizes = torch.stack([p.get_scaled_xy_span().detach().cpu() for h in range(len(det.hodoscopes)) for p in det.hodoscopes[h].panels], dim=0) / 2
                poss = torch.stack([p.xy.detach().cpu() for h in range(len(det.hodoscopes)) for p in det.hodoscopes[h].panels], dim=0)
            else:
                raise ValueError(f"Detector {det} is not supported")

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
