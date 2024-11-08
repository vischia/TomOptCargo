from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tomopt.volume.panel import DetectorPanel, SigmoidDetectorPanel
from tomopt.muon import MuonBatch
from tomopt.core import DEVICE


r"""
Provides implementations of class simulating panel-style detectors designed to be used for the Hodoscope class.
"""


__all__ = ["HodoscopeDetectorPanel", "HodoscopeSigmoidDetectorPanel"]


class HodoscopeDetectorPanel:
    r"""
    Provides an infinitely thin, rectangular panel in the xy plane, centred at the position of its parent Hodoscope (metres, in absolute position in the volume frame),
    """

    def __init__(
        self,
        *,
        res: float,
        eff: float,
        init_xyz: Tuple[float, float, float],
        init_xy_span: Tuple[float, float],
        m2_cost: float = 1,
        budget: Optional[Tensor] = None,
        realistic_validation: bool = True,
        device: torch.device = DEVICE,
        hod, #specifying type as 'Hodoscope' causes cyclic import error
        idx: int
    ):
        self.realistic_validation, self.device = realistic_validation, device
        self.xy_span = torch.tensor(init_xy_span, device=self.device)
        self.budget_scale = torch.ones(1, device=device)
        self.assign_budget(budget)
        self.m2_cost = 1.
        self.res = res
        self.eff = eff
        self.hod = hod
        self.idx = idx
       

    def __repr__(self) -> str:
        return f"""{self.__class__} located at xy={self.xy.data}, z={self.z.data}, and xy span {self.get_scaled_xy_span().data} with budget scale {self.budget_scale.data}"""

    def get_scaled_xy_span(self) -> Tensor:
        r"""
        Computes the effective size of the panel by rescaling based on the xy-span, cost per m^2, and budget.

        Returns:
            Rescaled xy-span such that the panel has a cost equal to the specified budget
        """

        return self.xy_span * self.budget_scale

    def get_xy_mask(self, xy: Tensor) -> Tensor:
        r"""
        Computes which of the xy points lie inside the physical panel.

        Arguments:
            xy: xy2) tensor of points

        Returns:
            (N,) Boolean mask, where True indicates the point lies inside the panel
        """

        span = self.get_scaled_xy_span()
        xy_low = self.xy - (span / 2)
        xy_high = self.xy + (span / 2)
        return (xy[:, 0] >= xy_low[0]) * (xy[:, 0] < xy_high[0]) * (xy[:, 1] >= xy_low[1]) * (xy[:, 1] < xy_high[1])

    def get_gauss(self) -> torch.distributions.Normal:
        r"""
        Returns:
            A Gaussian distribution, with 2 uncorrelated components corresponding to x and y, centred at the xy position of the panel, and sigma = panel span/4
        """

        try:
            return torch.distributions.Normal(self.xy, self.get_scaled_xy_span() / 4)  # We say that the panel widths corresponds to 2-sigma of the Gaussian
        except ValueError:
            raise ValueError(f"Invalid parameters for Gaussian: loc={self.xy}, scale={self.get_scaled_xy_span() / 4}")

    def get_resolution(self, xy: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        r"""
        Computes the xy resolutions of panel at the supplied list of xy points.
        If running in evaluation mode with `realistic_validation`,
        then these will be the full resolution of the panel for points inside the panel (indicated by the mask), and zero outside.
        Otherwise, the Gaussian model will be used.

        Arguments:
            xy: (N,xy) tensor of positions
            mask: optional pre-computed (N,) Boolean mask, where True indicates that the xy point is inside the panel.
                Only used in evaluation mode and if `realistic_validation` is True.
                If required, but not supplied, than will be computed automatically.

        Returns:
            res, a (N,xy) tensor of the resolution at the xy points
        """

        if not isinstance(self.res, Tensor):
            raise ValueError(f"{self.res} is not a Tensor for some reason.")  # To appease MyPy
        if self.hod.training or not self.realistic_validation:
            g = self.get_gauss()
            res = self.res * torch.exp(g.log_prob(xy)) / torch.exp(g.log_prob(self.xy))
            res = torch.clamp_min(res, 1e-10)  # To avoid NaN gradients
        else:
            if mask is None:
                mask = self.get_xy_mask(xy)
            res = torch.zeros((len(xy), 2), device=self.device)  # Zero detection outside detector
            res[mask] = self.res
        return res

    def get_efficiency(self, xy: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        r"""
        Computes the efficiency of panel at the supplied list of xy points.
        If running in evaluation mode with `realistic_validation`,
        then these will be the full efficiency of the panel for points inside the panel (indicated by the mask), and zero outside.
        Otherwise, the Gaussian model will be used.

        Arguments:
            xy: (N,) or (N,xy) tensor of positions
            mask: optional pre-computed (N,) Boolean mask, where True indicates that the xy point is inside the panel.
                Only used in evaluation mode and if `realistic_validation` is True.
                If required, but not supplied, than will be computed automatically.

        Returns:
            eff, a (N,)tensor of the efficiency at the xy points
        """

        if not isinstance(self.eff, Tensor):
            raise ValueError(f"{self.eff} is not a Tensor for some reason.")  # To appease MyPy
        if self.hod.training or not self.realistic_validation:
            g = self.get_gauss()
            scale = (torch.exp(g.log_prob(xy)) / torch.exp(g.log_prob(self.xy))).prod(dim=-1)
            eff = self.eff * scale
        else:
            if mask is None:
                mask = self.get_xy_mask(xy)
            eff = torch.zeros(len(xy), device=self.device)  # Zero detection outside detector
            eff[mask] = self.eff
        return eff

    def assign_budget(self, budget: Optional[Tensor] = None) -> None:
        r"""
        Sets the budget for the panel. This is then used to set a multiplicative coefficient, `budget_scale`, based on the `m2_cost`
        which rescales the `xy_span` such that the area of the resulting panel matches the assigned budget.

        Arguments:
            budget: required cost of the panel in unit currency
        """

        if budget is not None:
            self.budget_scale = torch.sqrt(budget / (self.m2_cost * self.xy_span.prod()))

    def get_hits(self, mu: MuonBatch) -> Dict[str, Tensor]:
        r"""
        The main interaction method with the panel: returns hits for the supplied muons.
        Hits consist of:
            reco_xy: (muons,xy) tensor of reconstructed xy positions of muons included simulated resolution
            gen_xy: (muons,xy) tensor of generator-level (true) xy positions of muons
            z: z position of the panel

        If running in evaluation mode with `realistic_validation`,
        then these will be the full resolution of the panel for points inside the panel (indicated by the mask), and zero outside.
        Otherwise, the Gaussian model will be used.
        """

        span = self.get_scaled_xy_span()
        mask = mu.get_xy_mask(self.xy - (span / 2), self.xy + (span / 2))  # Muons in panel
        true_mu_xy = mu.xy.data
        xy0 = self.xy - (span / 2)  # Low-left of panel
        rel_xy = true_mu_xy - xy0
        res = self.get_resolution(true_mu_xy, mask)
        rel_xy = rel_xy + (torch.randn((len(mu), 2), device=self.device) / res)

        if not self.hod.training and self.realistic_validation:  # Prevent reco hit from exiting panel
            np_span = span.detach().cpu().numpy()
            rel_xy[mask] = torch.stack([torch.clamp(rel_xy[mask][:, 0], 0, np_span[0]), torch.clamp(rel_xy[mask][:, 1], 0, np_span[1])], dim=-1)
        reco_xy = xy0 + rel_xy

        reco_xyz = F.pad(reco_xy, (0, 1))
        reco_xyz[:, 2] = self.z
        gen_xyz = F.pad(true_mu_xy, (0, 1))
        gen_xyz[:, 2] = self.z
        hits = {
            "reco_xyz": reco_xyz,
            "gen_xyz": gen_xyz,
            "unc_xyz": F.pad(1 / res, (0, 1)),  # Add zero for z unc
            "eff": self.get_efficiency(true_mu_xy, mask)[:, None],
        }

        return hits

    def get_cost(self) -> Tensor:
        r"""
        Returns:
            current cost of the panel according to its area and m2_cost
        """

        return self.m2_cost * self.get_scaled_xy_span().prod()

    def clamp_params(self, xyz_low: Tuple[float, float, float], xyz_high: Tuple[float, float, float]) -> None:
        r"""
        Ensures that the panel is centred within the supplied xyz range,
        and that the span of the panel is between xyz_high/20 and xyz_high*10.
        A small random number < 1e-3 is added/subtracted to the min/max z position of the panel, to ensure it doesn't overlap with other panels.

        Arguments:
            xyz_low: minimum x,y,z values for the panel centre in metres
            xyz_high: maximum x,y,z values for the panel centre in metres
        """

        with torch.no_grad():
            eps = np.random.uniform(0, 1e-3)  # prevent hits at same z due to clamping
            self.x.clamp_(min=xyz_low[0], max=xyz_high[0])
            self.y.clamp_(min=xyz_low[1], max=xyz_high[1])
            self.z.clamp_(min=xyz_low[2] + eps, max=xyz_high[2] - eps)
            self.xy_span[0].clamp_(min=xyz_high[0] / 20, max=10 * xyz_high[0])
            self.xy_span[1].clamp_(min=xyz_high[1] / 20, max=10 * xyz_high[1])

    def forward(self) -> None:
        raise NotImplementedError("Please do not use forward, instead use get_hits")

    def get_z_in_hod(self) -> Tensor:
        r"""
        Get the z position of the panel within the hodoscope.
        """
        return self.hod.z - self.hod.xyz_gap[2] - (self.hod.xyz_span[2] - 2 * self.hod.xyz_gap[2]) * self.idx / (self.hod.n_panels - 1)

    @property
    def x(self) -> Tensor:
        return self.hod.xy[0]

    @property
    def y(self) -> Tensor:
        return self.hod.xy[1]
    
    @property
    def z(self) -> Tensor:
        return  self.get_z_in_hod() 

    @property
    def xy(self) -> Tensor:
        return self.hod.xy

class HodoscopeSigmoidDetectorPanel(HodoscopeDetectorPanel):

    def __init__(
        self,
        *,
        res: float,
        eff: float,
        init_xyz: Tuple[float, float, float],
        init_xy_span: Tuple[float, float],
        m2_cost: float = 1,
        budget: Optional[Tensor] = None,
        realistic_validation: bool = True,
        device: torch.device = DEVICE,
        hod:Any,
        idx: int,
        smooth:Union[float, Tensor],

    ):
        super().__init__(
            res=res,
            eff=eff,
            init_xyz=init_xyz,
            init_xy_span=init_xy_span,
            m2_cost=m2_cost,
            budget=budget,
            realistic_validation=realistic_validation,
            device=device,
            hod = hod,
            idx = idx
            
        )
        self.smooth = smooth
        
        

    def sig_model(self, xy: Tensor) -> Tensor:
        r"""
        Models fractional resolution and efficiency from a sigmoid-based model to provide a smooth and differentiable model of a physical detector-panel.

        Arguments:
            xy: (N,xy) tensor of positions

        Returns:
            Multiplicative coefficients for the nominal resolution or efficiency of the panel based on the xy position relative to the panel position and size
        """

        half_width = self.get_scaled_xy_span() / 2
        delta = (xy - self.xy) / half_width
        coef = torch.sigmoid((1 - (torch.sign(delta) * delta)) / self.smooth)
        return coef / torch.sigmoid(torch.tensor(1 / self.smooth))

    def get_resolution(self, xy: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        r"""
        Computes the xy resolutions of panel at the supplied list of xy points.
        If running in evaluation mode with `realistic_validation`,
        then these will be the full resolution of the panel for points inside the panel (indicated by the mask), and zero outside.
        Otherwise, the Sigmoid model will be used.

        Arguments:
            xy: (N,xy) tensor of positions
            mask: optional pre-computed (N,) Boolean mask, where True indicates that the xy point is inside the panel.
                Only used in evaluation mode and if `realistic_validation` is True.
                If required, but not supplied, than will be computed automatically.

        Returns:
            res, a (N,xy) tensor of the resolution at the xy points
        """
        if self.hod.training or not self.realistic_validation:
            res = self.res * self.sig_model(xy)
            res = torch.clamp_min(res, 1e-10)  # To avoid NaN gradients
        else:
            if mask is None:
                mask = self.get_xy_mask(xy)
            res = torch.zeros((len(xy), 2), device=self.device)  # Zero detection outside detector
            res[mask] = self.res
        return res

    def get_efficiency(self, xy: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        r"""
        Computes the efficiency of panel at the supplied list of xy points.
        If running in evaluation mode with `realistic_validation`,
        then these will be the full efficiency of the panel for points inside the panel (indicated by the mask), and zero outside.
        Otherwise, the Sigmoid model will be used.

        Arguments:
            xy: (N,) or (N,xy) tensor of positions
            mask: optional pre-computed (N,) Boolean mask, where True indicates that the xy point is inside the panel.
                Only used in evaluation mode and if `realistic_validation` is True.
                If required, but not supplied, than will be computed automatically.

        Returns:
            eff, a (N,)tensor of the efficiency at the xy points
        """

        if self.hod.training or not self.realistic_validation:
            eff = self.eff * self.sig_model(xy).prod(dim=-1)
            eff = torch.clamp_min(eff, 1e-10)  # To avoid NaN gradients
        else:
            if mask is None:
                mask = self.get_xy_mask(xy)
            eff = torch.zeros(len(xy), device=self.device)  # Zero detection outside detector
            eff[mask] = self.eff
        return eff
        
    def get_z_in_hod(self) -> Tensor:
        r"""
        Get the z position of the panel within the hodoscope.
        """
        return self.hod.z - self.hod.xyz_gap[2] - (self.hod.xyz_span[2] - 2 * self.hod.xyz_gap[2]) * self.idx / (self.hod.n_panels - 1)

    @property
    def x(self) -> Tensor:
        return self.hod.xy[0]

    @property
    def y(self) -> Tensor:
        return self.hod.xy[1]

    @property
    def z(self) -> Tensor:
        return  self.get_z_in_hod() 

    @property
    def xy(self) -> Tensor:
        return self.hod.xy