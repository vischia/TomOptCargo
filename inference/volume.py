from typing import Type

import torch
from torch import Tensor

from tomopt.volume import Volume
from tomopt.inference.scattering import ScatterBatch
from tomopt.inference.volume import AbsVolumeInferrer, AbsX0Inferrer, PanelX0Inferrer

r"""
Provides implementations of classes designed to infer targets of passive volumes
using the variables computed by e.g. :class:`~tomopt.inference.scattering.ScatterBatch`.
"""

__all__ = [
    "BCAVoxelInferrer"
    "BCAVoxelInferrerExtended",
]  

class BCAVoxelInferrer(AbsVolumeInferrer):

    r"""
    Class for implementing the Binned-Clustering_algorithm (BCA) (https://iopscience.iop.org/article/10.1088/1748-0221/8/10/P10013) to infer voxel-wise scores based on PoCA variables.
    This implementation makes use of reconstructed PoCA points. The algorithm loops over the voxels of the passive volume and selects the first n PoCA points in descending order of their scattering angles.
    For each voxel, a metric is computed, which is the distance between PoCA pairs, weighted by the product of their scattering angles. The original algorihm normalizes the scattering angles by the corresponding muon momenta.
    However, since momentum estimation is not currently implemented, this normalization is neglected.
    The voxel score is the median of the natural logarithm of the obtained distribution.

    .. important::
        In orde to have enough number of PoCA points inside all voxels to be able to perform the selection of the points with largest scattering angles, 
        the number of generated muons should be sufficiently increased, which significantlly increases computation time.

    Arguments:
        partial_x0_inferrer: (partial) class to instatiate to provide the PoCA variables
        volume: volume through which the muons will be passed
        n_max: optional maxumum number of PoCA points within a voxel to be selected for BCA algorithm
    """

    
    def __init__(
        self,
        volume: Volume,
        partial_x0_inferrer: Type[AbsX0Inferrer] = PanelX0Inferrer,
        n_max: int = 40,
    ):
        r"""
        Initialises the inference class for the provided volume.
        """

        super().__init__(volume=volume)
        self.n_max = n_max
        self.x0_inferrer = partial_x0_inferrer(volume=self.volume)


    def add_scatters(self, scatters: ScatterBatch) -> None:
        r"""
        Appends a new set of muon scatter vairables.
        When :meth:`~tomopt.inference.volume.DenseBlockClassifierFromX0s.get_prediction` is called, the prediction will be based on all
        :class:`~tomopt.inference.scattering.ScatterBatch` s added up to that point
        """

        self.x0_inferrer.add_scatters(scatters)

    def compute_efficiency(self, scatters: ScatterBatch) -> Tensor:
        r"""
        Compuates the per-muon efficiency according to the method implemented by the X0 inferrer.

        Arguments:
            scatters: scatter batch containing muons whose efficiency should be computed

        Returns:
            (muons) tensor of muon efficiencies
        """

        return self.x0_inferrer.compute_efficiency(scatters=scatters)

    def get_prediction(self) -> Tensor:
        r"""
        Computes the BCA score predictions per voxel using the scatter batches added.

        Returns:
            (z,x,y) tensor of voxelwise BCA predictions
        """
        # number of voxels in x, y and z directions
        nx, ny, nz = self.shp_xyz 

        bca_score = torch.zeros(self.shp_xyz)
        # assign a voxel index ID for PoCA points
        pocas_vox_id= torch.floor(self.x0_inferrer.muon_poca_xyz/self.size)
        pocas_vox_id [:, 2] = pocas_vox_id [:, 2] - torch.floor(self.volume.get_passive_z_range()[0]/self.size) # start z indexing at 0 by subtracting z index of lower passive layer
        
        # loop over voxels
        for i in range(nx):
            for j in range(ny):
                for k in range (nz):
                    # voxel (ijk)
                    voxel_id = torch.tensor([i, j, k])
                    # mask to find PoCA points that match this voxel ID
                    mask = torch.all(pocas_vox_id == voxel_id, dim=1)
                    # total scattering angles of PoCA points withn voxel (ijk)
                    tot_scatter = self.x0_inferrer.muon_total_scatter.squeeze(1)[mask]
                    # sort PoCA points in decreasing order of their total scattering angles
                    pocas_dtheta_sorted, idx = torch.sort(tot_scatter,descending=True)
                    # product of scattering angles of PoCA pairs of a selection of the first n_max PoCA's with highest scattering angles, to weight the BCA metric
                    dtheta_wgt = pocas_dtheta_sorted[:self.n_max].unsqueeze(1) * pocas_dtheta_sorted[:self.n_max].unsqueeze(0)
                    idx = idx[:self.n_max]
                    # calculating BCA metric using the n_max PoCA points inside voxel (ijk)
                    pocas_in_voxel = self.x0_inferrer.muon_poca_xyz[mask]
                    pocas_in_voxel_large_dtheta = pocas_in_voxel[idx]
                    difference = pocas_in_voxel_large_dtheta.unsqueeze(1) - pocas_in_voxel_large_dtheta.unsqueeze(0)
                    metric =  torch.norm(difference, dim=2)
                    # weighting wiyh product of scattering angles
                    weighted_metric = torch.tril(torch.where((metric!=0)&(dtheta_wgt!=0),metric/dtheta_wgt,0.))
                    # final score of voxel (ijk)
                    score = torch.log(weighted_metric[weighted_metric!=0]).median()
                    score = score.nan_to_num()
                    bca_score[i][j][k]+=score

        pred = bca_score

        if pred.isnan().any():
            raise ValueError("Prediction contains NaN values")

        return pred.permute(2,0,1) #z,x,y

    def _reset_vars(self) -> None:
        r"""
        Resets any variable/predictions made from the added scatter batches.
        """

        self.x0_inferrer._reset_vars()


class BCAVoxelInferrerExtended(AbsVolumeInferrer):

    r"""
    Class for implementing the Binned-Clustering_algorithm (BCA) (https://iopscience.iop.org/article/10.1088/1748-0221/8/10/P10013) to infer voxel-wise scores based on PoCA variables.
    This implementation makes use of reconstructed PoCA points. The algorithm loops over the voxels of the passive volume and calculates a metric, which is the distance between PoCA 
    pairs, weighted by the product of their scattering angles. The original algorihm normalizes the scattering angles by the corresponding muon momenta.
    However, since momentum estimation is not currently implemented, this normalization is neglected.
    This modified inferrer uses extended PoCA probability distributions to assign a per voxel probability of muons. This probability and the muon efficiency are used as weights to compute a weighted median for the voxel BCA score.
    No selection of PoCA points is applied at the voxel level, since all muons contribute with their weights. This removes the requirement of generating a very large number of muons.

    Arguments:
        partial_x0_inferrer: (partial) class to instatiate to provide the PoCA variables
        volume: volume through which the muons will be passed
    """
    
    def __init__(
        self,
        volume: Volume,
        partial_x0_inferrer: Type[AbsX0Inferrer] = PanelX0Inferrer,
    ):
        r"""
        Initialises the inference class for the provided volume.
        """

        super().__init__(volume=volume)
        self.x0_inferrer = partial_x0_inferrer(volume=self.volume)

    def add_scatters(self, scatters: ScatterBatch) -> None:
        r"""
        Appends a new set of muon scatter vairables.
        When :meth:`~tomopt.inference.volume.DenseBlockClassifierFromX0s.get_prediction` is called, the prediction will be based on all
        :class:`~tomopt.inference.scattering.ScatterBatch` s added up to that point
        """

        self.x0_inferrer.add_scatters(scatters)

    def compute_efficiency(self, scatters: ScatterBatch) -> Tensor:
        r"""
        Compuates the per-muon efficiency according to the method implemented by the X0 inferrer.

        Arguments:
            scatters: scatter batch containing muons whose efficiency should be computed

        Returns:
            (muons) tensor of muon efficiencies
        """

        return self.x0_inferrer.compute_efficiency(scatters=scatters)

    def get_prediction(self) -> Tensor:
        r"""
        Computes the BCA score predictions per voxel using the scatter batches added.

        Returns:
            (z,x,y) tensor of voxelwise BCA predictions
        """
        # number of voxels in x, y and z directions
        nx, ny, nz = self.shp_xyz 
        #metric
        difference = self.x0_inferrer.muon_poca_xyz.unsqueeze(1) - self.x0_inferrer.muon_poca_xyz.unsqueeze(0)
        metric =  torch.norm(difference, dim=2)
        #weighting by scattering angles
        dtheta_wgt =self.x0_inferrer.muon_total_scatter.squeeze(1).unsqueeze(1) * self.x0_inferrer.muon_total_scatter.squeeze(1).unsqueeze(0)
        #non-zero scatter angles in denominator
        non_zero_mask = (dtheta_wgt != 0)
        metric=metric.clone()
        metric[non_zero_mask] = metric[non_zero_mask] / dtheta_wgt[non_zero_mask]
        metric =torch.tril(metric)
        
        muon_prob = self.x0_inferrer.muon_probs_per_voxel_zxy
        bca_score = torch.zeros(self.shp_xyz)
        # assign a voxel index ID for PoCA points
        pocas_vox_id= torch.floor(self.x0_inferrer.muon_poca_xyz/self.size)
        pocas_vox_id [:, 2] = pocas_vox_id [:, 2] - torch.floor(self.volume.get_passive_z_range()[0]/self.size) # start z indexing at 0 by subtracting z index of lower passive layer
        
        # loop over voxels
        for i in range(nx):
            for j in range(ny):
                for k in range (nz):
                    # voxel (ijk)
                    #muon weight = per voxel prob * efficiency
                    muon_wgt = torch.tril((self.x0_inferrer.muon_efficiency* muon_prob[:,k,i,j]).unsqueeze(1) * (muon_prob[:,k,i,j]*self.x0_inferrer.muon_efficiency).unsqueeze(0))
                    mask = (metric!=0) & (muon_wgt!=0)
                    metric_masked = torch.log(metric[mask])
                    muon_wgt = muon_wgt[mask]
                    #median weighted by the muon weights
                    score = self.weighted_median(metric_masked, muon_wgt)
                    score = score.nan_to_num()
                    bca_score[i][j][k]+=score

        pred = bca_score

        if pred.isnan().any():
            raise ValueError("Prediction contains NaN values")

        return pred.permute(2,0,1)
    
    
    def weighted_median(self, data: Tensor, unc: Tensor) -> Tensor:
    
        # Sort data and uncertainties
        sorted_data, indices = torch.sort(data)
        sorted_uncertainties = unc[indices]

        # Normalize uncertainties (weights)
        weights = sorted_uncertainties / sorted_uncertainties.sum()

        # Compute cumulative sum of weights
        cumulative_weights = torch.cumsum(weights, dim = 0)
    
        # Find the index where cumulative weight first crosses 0.5
        median_idx = torch.where(cumulative_weights >= 0.5)[0][0]
        
        # The weighted median is the corresponding data point
        weighted_median = sorted_data[median_idx]

        return weighted_median

    def _reset_vars(self) -> None:
        r"""
        Resets any variable/predictions made from the added scatter batches.
        """

        self.x0_inferrer._reset_vars()


    


