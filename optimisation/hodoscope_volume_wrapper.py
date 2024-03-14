
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from fastcore.all import Path
from fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import ConsoleProgressBar, NBProgressBar, ProgressBar
from torch import Tensor
from torch.optim.optimizer import Optimizer

from tomopt.core import DEVICE, PartialOpt
from tomopt.inference import AbsVolumeInferrer, PanelX0Inferrer, ScatterBatch
from tomopt.muon import AbsMuonGenerator, MuonBatch, MuonGenerator2016
from tomopt.optimisation.loss.loss import AbsDetectorLoss
from tomopt.volume import AbsDetectorLayer, PanelDetectorLayer, Volume
from tomopt.optimisation.callbacks import (
    Callback,
    CyclicCallback,
    EvalMetric,
    MetricLogger,
    PredHandler,
    WarmupCallback,
)
from tomopt.optimisation.data import PassiveYielder
from tomopt.optimisation.wrapper import FitParams, AbsVolumeWrapper
from volume.hodoscopelayer import HodoscopeDetectorLayer

class HodoscopeVolumeWrapper(AbsVolumeWrapper):
    r"""
    Volume wrapper for volumes with :class:`~tomopt.volume.panel.DetectorPanel`-based detectors.

    Volume wrappers are designed to contain a :class:`~tomopt.volume.volume.Volume` and provide means of optimising the detectors it contains,
    via their :meth:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper.fit` method.

    Wrappers also provide for various quality-of-life methods, such as saving and loading detector configurations,
    and computing predictions with a fixed detector (:meth:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper.predict`)

    Fitting of a detector proceeds as training and validation epochs, each of which contains multiple batches of passive volumes.
    For each volume in a batch, the loss is evaluated using multiple batches of muons.
    The whole loop is:

    1. for epoch in `n_epochs`:
        A. `loss` = 0
        B. for `p`, `passive` in enumerate(`trn_passives`):
            a. load `passive` into passive volume
            b. for muon_batch in range(`n_mu_per_volume`//`mu_bs`):
                i. Irradiate volume with `mu_bs` muons
                ii. Infer passive volume
            c. Compute loss based on precision and cost, and add to `loss`
            d. if `p`+1 % `passive_bs` == 0:
                i. `loss` = `loss`/`passive_bs`
                ii. Backpropagate `loss` and update detector parameters
                iii. `loss` = 0
            e. if len(`trn_passives`)-(`p`+1) < `passive_bs`:
                i. Break

        C. `val_loss` = 0
        D. for `p`, `passive` in enumerate(`val_passives`):
            a. load `passive` into passive volume
            b. for muon_batch in range(`n_mu_per_volume`//`mu_bs`):
                i. Irradiate volume with `mu_bs` muons
                ii. Infer passive volume
                iii. Compute loss based on precision and cost, and add to `val_loss`
            c. if len(`val_passives`)-(`p`+1) < `passive_bs`:
                i. Break
        E. `val_loss` = `val_loss`/`p`

    In implementation, the loop is broken up into several functions:
        :meth:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper._fit_epoch` runs one full epoch of volumes
            and updates for both training and validation
        :meth:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper._scan_volumes` runs over all training/validation volumes,
            updating parameters when necessary
        :meth:`~tomopt.optimisation.wrapper.volume_wrapper.AbsVolumeWrapper._scan_volume` irradiates a single volume with muons multiple batches,
            and computes the loss for that volume

    The optimisation and prediction loops are supported by a stateful callback mechanism.
    The base callback is :class:`~tomopt.optimisation.callbacks.callback.Callback`, which can interject at various points in the loops.
    All aspects of the optimisation and prediction are stored in a :class:`~tomopt.optimisation.wrapper.volume_wrapper.FitParams` data class,
    since the callbacks are also stored there, and the callbacks have a reference to the wrapper, they are able to read/write to the `FitParams` and be
    aware of other callbacks that are running.

    Accounting for the interjection calls (`on_*_begin` & `on_*_end`), the full optimisation loop is:

    1. Associate callbacks with wrapper (`set_wrapper`)
    2. `on_train_begin`
    3. for epoch in `n_epochs`:
        A. `state` = "train"
        B. `on_epoch_begin`
        C. for `p`, `passive` in enumerate(`trn_passives`):
            a. if `p` % `passive_bs` == 0:
                i. `on_volume_batch_begin`
                ii. `loss` = 0
            b. load `passive` into passive volume
            c. `on_volume_begin`
            d. for muon_batch in range(`n_mu_per_volume`//`mu_bs`):
                i. `on_mu_batch_begin`
                ii. Irradiate volume with `mu_bs` muons
                iii. Infer scatter locations
                iv. `on_scatter_end`
                v. Infer x0 and append to list of x0 predictions
                vi. `on_mu_batch_end`
            e. `on_x0_pred_begin`
            f. Compute overall x0 prediction
            g. `on_x0_pred_end`
            h. Compute loss based on precision and cost, and add to `loss`
            i. if `p`+1 % `passive_bs` == 0:
                i. `loss` = `loss`/`passive_bs`
                ii. `on_volume_batch_end`
                iii. Zero parameter gradients
                iv. `on_backwards_begin`
                v. Backpropagate `loss` and compute parameter gradients
                vi. `on_backwards_end`
                vii. Update detector parameters
                viii. Ensure detector parameters are within physical boundaries (`AbsDetectorLayer.conform_detector`)
                viv. `loss` = 0
            j. if len(`trn_passives`)-(`p`+1) < `passive_bs`:
                i. Break
        D. `on_epoch_end`
        E. `state` = "valid"
        F. `on_epoch_begin`
        G. for `p`, `passive` in enumerate(`val_passives`):
            a. if `p` % `passive_bs` == 0:
                i. `on_volume_batch_begin`
                ii. `loss` = 0
            b. `on_volume_begin`
            c. for muon_batch in range(`n_mu_per_volume`//`mu_bs`):
                i. `on_mu_batch_begin`
                ii. Irradiate volume with `mu_bs` muons
                iii. Infer scatter locations
                iv. `on_scatter_end`
                v. Infer x0 and append to list of x0 predictions
                vi. `on_mu_batch_end`
            d. `on_x0_pred_begin`
            e. Compute overall x0 prediction
            f. `on_x0_pred_end`
            g. Compute loss based on precision and cost, and add to `loss`
            h. if `p`+1 % `passive_bs` == 0:
                i. `loss` = `loss`/`passive_bs`
                ii. `on_volume_batch_end`
            i. if len(`val_passives`)-(`p`+1) < `passive_bs`:
                i. Break
        H. `on_epoch_end`
    4. `on_train_end`

    Arguments:
        volume: the volume containing the detectors to be optimised
        xy_pos_opt: uninitialised optimiser to be used for adjusting the xy position of panels
        z_pos_opt: uninitialised optimiser to be used for adjusting the z position of panels
        xy_span_opt: uninitialised optimiser to be used for adjusting the xy size of panels
        budget_opt: optional uninitialised optimiser to be used for adjusting the fractional assignment of budget to the panels
        loss_func: optional loss function (required if planning to optimise the detectors)
        partial_scatter_inferrer: uninitialised class to be used for inferring muon scatter variables and trajectories
        partial_volume_inferrer:  uninitialised class to be used for inferring volume targets
        mu_generator: Optional generator class for muons. If None, will use :meth:`~tomopt.muon.generation. MuonGenerator2016.from_volume`.
    """

    def __init__(
        self,
        volume: Volume,
        *,
        xy_pos_opt: PartialOpt,
        z_pos_opt: PartialOpt,
        xyz_span_opt: PartialOpt,
        budget_opt: Optional[PartialOpt] = None,
        loss_func: Optional[AbsDetectorLoss] = None,
        partial_scatter_inferrer: Type[ScatterBatch] = ScatterBatch,
        partial_volume_inferrer: Type[AbsVolumeInferrer] = PanelX0Inferrer,
        mu_generator: Optional[AbsMuonGenerator] = None,
    ):
        super().__init__(
            volume=volume,
            partial_opts={
                "xy_pos_opt": xy_pos_opt,
                "z_pos_opt": z_pos_opt,
                "xy_span_opt": xyz_span_opt,
                "budget_opt": budget_opt,
            },
            loss_func=loss_func,
            mu_generator=mu_generator,
            partial_scatter_inferrer=partial_scatter_inferrer,
            partial_volume_inferrer=partial_volume_inferrer,
        )

    @classmethod
    def from_save(
        cls,
        name: str,
        *,
        volume: Volume,
        xy_pos_opt: PartialOpt,
        z_pos_opt: PartialOpt,
        xy_span_opt: PartialOpt,
        budget_opt: Optional[PartialOpt] = None,
        loss_func: Optional[AbsDetectorLoss],
        partial_scatter_inferrer: Type[ScatterBatch] = ScatterBatch,
        partial_volume_inferrer: Type[AbsVolumeInferrer] = PanelX0Inferrer,
        mu_generator: Optional[AbsMuonGenerator] = None,
    ) -> AbsVolumeWrapper:
        r"""
        Instantiates a new `PanelVolumeWrapper` and loads saved detector and optimiser parameters

        Arguments:
            name: file name with saved detector and optimiser parameters
            volume: the volume containing the detectors to be optimised
            xy_pos_opt: uninitialised optimiser to be used for adjusting the xy position of panels
            z_pos_opt: uninitialised optimiser to be used for adjusting the z position of panels,
            xy_span_opt: uninitialised optimiser to be used for adjusting the xy size of panels,
            budget_opt: optional uninitialised optimiser to be used for adjusting the fractional assignment of budget to the panels
            loss_func: optional loss function (required if planning to optimise the detectors)
            partial_scatter_inferrer: uninitialised class to be used for inferring muon scatter variables and trajectories
            partial_volume_inferrer:  uninitialised class to be used for inferring volume targets
            mu_generator: Optional generator class for muons. If None, will use :meth:`~tomopt.muon.generation. MuonGenerator2016.from_volume`.
        """

        vw = cls(
            volume=volume,
            xy_pos_opt=xy_pos_opt,
            z_pos_opt=z_pos_opt,
            xy_span_opt=xy_span_opt,
            budget_opt=budget_opt,
            loss_func=loss_func,
            partial_scatter_inferrer=partial_scatter_inferrer,
            partial_volume_inferrer=partial_volume_inferrer,
            mu_generator=mu_generator,
        )
        vw.load(name)
        return vw

    def _build_opt(self, **kwargs: PartialOpt) -> None:
        r"""
        Initialises the optimisers by associating them to the detector parameters.

        Arguments:
            kwargs: uninitialised optimisers passed as keyword arguments
        """

        all_dets = self.volume.get_detectors()
        dets: List[HodoscopeDetectorLayer] = []

        for d in all_dets:
            # if isinstance(d, PanelDetectorLayer):
            if isinstance(d, AbsDetectorLayer):

                dets.append(d)
        self.opts = {
            "xy_pos_opt": kwargs["xy_pos_opt"]((h.xy for l in dets for h in l.hodoscopes)),
            "z_pos_opt": kwargs["z_pos_opt"]((h.z for l in dets for h in l.hodoscopes)),
            "xyz_span_opt": kwargs["z_pos_opt"]((h.xyz_span for l in dets for h in l.hodoscopes)),
        }
        if kwargs["budget_opt"] is not None:
            self.opts["budget_opt"] = kwargs["budget_opt"]((p for p in [self.volume.budget_weights]))