import matplotlib.pyplot as plt
import numpy as np
import torch

# TomOptCargo
from hodoscopelayer import HodoscopeDetectorLayer

# OG tomopt
from tomopt.volume.volume import Volume
from tomopt.muon.muon_batch import MuonBatch
from typing import Tuple


def get_panels_xy_min_max(volume: Volume) -> Tuple[float, float, float, float]:
    """
        Returns xmin, xmax, ymin, ymax, the min max x and y coordinates of the edges of the volume's panels.
        Useful for plotting purposes.
    """

    xys = np.array([p.xy.detach().numpy() for l in volume.layers if isinstance(l, HodoscopeDetectorLayer) for i,p in l.yield_zordered_panels()])
    xy_spans = np.array([p.xy_span.detach().numpy() for l in volume.layers if isinstance(l, HodoscopeDetectorLayer) for i,p in l.yield_zordered_panels()])

    xy_mins, xy_maxs = xys - xy_spans/2, xys + xy_spans/2 

    xmin, ymin = np.min([xy[0] for xy in xy_mins]), np.min([xy[1] for xy in xy_mins])
    xmax, ymax = np.max([xy[0] for xy in xy_maxs]), np.max([xy[1] for xy in xy_maxs])

    return xmin, xmax, ymin, ymax

def get_panels_z_min_max(volume: Volume) -> Tuple[float, float]:
    r"""
        Returns zmin and zmax, the min max z coordinates of the volume's panels.
        Useful for plotting purposes.
    """
    zs = [p.z.detach().item() for l in volume.layers if isinstance(l, HodoscopeDetectorLayer) for i,p in l.yield_zordered_panels()]
    return np.min(zs), np.max(zs)

def plot_hits(volume: Volume, 
              muons: MuonBatch, 
              event: int = 0, 
              dim: int = 0, 
              gap: float = .1, 
              hits: str = 'reco_xyz') -> None:

    r"""
    Plots the volumes PanelDetectorLayers and the recorded hits.

    Arguments:
            - volume: Instance of the Volume class.
            - muons: Instance of the MuonBatch class.
            - event: int, The muon event for which to plot hits.
            - dim: int. Plot hits as xz projection if dim==0, as yz if dim==1.
            - gap: float. Used to set the ax limits. Distance between outmost left/right panel
            and the ax left/right lim.
            - hits: str, the hits to plot. Can be either 'reco_xyz' or 'gen_xyz'.

    Temporary method, used for debuging purposes.
    """
    assert len(muons._hits) > 0, 'Input Muonbatch has not recorded any hits'

    fig, ax = plt.subplots()

    # set plot lim
    xy_min_max = get_panels_xy_min_max(volume)
    zmin, zmax = get_panels_z_min_max(volume)
    ax.set(xlim = [xy_min_max[dim] - gap, xy_min_max[dim + 1] + gap], ylim = [zmin - gap, zmax + gap])
    
    # if extended_view:
    #     xyz_hits = np.asarray([muons._hits[l.pos]['reco_xyz'][i][event].detach().numpy() for l in volume.layers if isinstance(l, HodoscopeDetectorLayer) for i,p in l.yield_zordered_panels()])
    #     xmin, xmax = np.min(xyz_hits[:,dim]), np.max(xyz_hits[:,dim])
    #     ax.set(xlim = [xmin - gap, xmax + gap], ylim = [zmin - gap, zmax + gap])

    # axis label
    xlabel = 'x [m]' if dim == 0 else 'y [m]'
    ax.set_ylabel("z [m]")
    ax.set_xlabel(xlabel)

    # get relative panels pos
    def relative_pos(ax, x: float) -> float:
        return (x - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])

    for l in volume.layers:
        if isinstance(l, HodoscopeDetectorLayer):
            for i, p in l.yield_zordered_panels():
                xyz = muons._hits[l.pos][hits][i][event].detach().numpy()
                # plot legend only once
                legend = hits if (i == 0) & (l.pos == 'above') else None
                # plot reco efficiency if reco_hits
                # alpha =  muons._hits[l.pos]['eff'][i][event].detach().item() if hits == 'reco_xyz' else .9
                alpha =  .9
                # plot hits
                ax.scatter(xyz[dim], xyz[2], color = "blue", label = legend, alpha = alpha)
                # plot legend only once
                legend = 'panels' if (i == 0) & (l.pos == 'above') else None
                # plot panels
                ax.axhline(y = p.z.detach().numpy(),
                           color = "red",
                           alpha = .6,
                           xmin = relative_pos(ax, (p.xy[dim] - p.xy_span[0] / 2).detach().item()), 
                           xmax = relative_pos(ax, (p.xy[dim] + p.xy_span[0] / 2).detach().item()),
                           label = legend
                            )


    ax.set_aspect('equal')
    ax.legend()
    plt.show()