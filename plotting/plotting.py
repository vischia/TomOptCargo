import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from typing import Tuple, Optional
from torch import Tensor

# TomOptCargo
from volume.hodoscopelayer import HodoscopeDetectorLayer

# OG tomopt
from tomopt.volume.volume import Volume
from tomopt.volume.layer import PassiveLayer
from tomopt.muon.muon_batch import MuonBatch




def get_passives_xy_min_max(volume: Volume) -> np.ndarray:
    """
        Returns xmin, xmax, ymin, ymax, the min max x and y coordinates of the edges of the volume's panels.
        Useful for plotting purposes.
    """
    xys = np.array([l.lw.numpy() for l in volume.layers if isinstance(l, PassiveLayer)])
    xmax, ymax = np.max(xys[:, 0]), np.max(xys[:, 1])

    return np.array([0., xmax, 0., ymax])

def get_panels_xy_min_max(volume: Volume) -> np.ndarray:
    """
        Returns xmin, xmax, ymin, ymax, the min max x and y coordinates of the edges of the volume's panels.
        Useful for plotting purposes.
    """

    xys = np.array([p.xy.detach().numpy() for l in volume.layers if isinstance(l, HodoscopeDetectorLayer) for i,p in l.yield_zordered_panels()])
    xy_spans = np.array([p.xy_span.detach().numpy() for l in volume.layers if isinstance(l, HodoscopeDetectorLayer) for i,p in l.yield_zordered_panels()])

    xy_mins, xy_maxs = xys - xy_spans/2, xys + xy_spans/2 

    xmin, ymin = np.min([xy[0] for xy in xy_mins]), np.min([xy[1] for xy in xy_mins])
    xmax, ymax = np.max([xy[0] for xy in xy_maxs]), np.max([xy[1] for xy in xy_maxs])

    return np.array([xmin, xmax, ymin, ymax])

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

def draw_volume_2D(volume: Volume, hits: Optional[Tensor] = None, pocas: Optional[Tensor] = None, event: Optional[int] = None) -> None:

    fig, axs = plt.subplots(ncols=3, figsize = (10, 10))

    gap = 0.3 # m

    # min max coord of hodoscope layers
    xy_min_max_panels = get_panels_xy_min_max(volume)

    # min max coord of passive layers
    xy_min_max_passives = get_passives_xy_min_max(volume)

    xy_min = np.where(xy_min_max_panels < xy_min_max_passives, xy_min_max_panels, xy_min_max_passives)
    xy_max = np.where(xy_min_max_panels > xy_min_max_passives, xy_min_max_panels, xy_min_max_passives)

    # z min max hodoscopes
    z_min_max = get_panels_z_min_max(volume)

    # set axis lims
    axs[0].set_xlim([xy_min[0] - gap, xy_max[1] + gap])
    axs[0].set_ylim([z_min_max[0]- gap, z_min_max[1] + gap])

    axs[1].set_xlim([xy_min[2]- gap, xy_max[3] + gap])
    axs[1].set_ylim([z_min_max[0]- gap, z_min_max[1] + gap])

    axs[2].set_xlim([xy_min[0]- gap, xy_max[1] + gap])
    axs[2].set_ylim([xy_min[2]- gap, xy_max[3] + gap])

    for ax in axs:
        ax.set_aspect("equal")
        ax.grid("on")

    hods, panels = [], []

    for l in volume.layers:
        if isinstance(l, HodoscopeDetectorLayer):
            for h in l.hodoscopes:
                hods.append({"x": h.xy[0].detach().item() - h.xyz_span[0].detach().item() / 2, 
                            "y": h.xy[1].detach().item() - h.xyz_span[1].detach().item() / 2,
                            "z": h.z.detach().item() - h.xyz_span[2].detach().item(), 
                            "dx": h.xyz_span[0].detach().item(), 
                            "dy": h.xyz_span[1].detach().item(), 
                            "dz": h.xyz_span[2].detach().item()})
                for p in h.panels:
                    panels.append({"x": p.xy[0].detach().item(), 
                                    "y": p.xy[1].detach().item(), 
                                    "z": p.z.detach().item(), 
                                    "dx": p.xy_span[0].detach().item(),
                                    "dy": p.xy_span[1].detach().item()})

    # XZ view
    XZ_hods = [Rectangle((hod["x"], hod["z"]), 
                            hod["dx"], 
                            hod["dz"]) for hod in hods]
    # plot hodoscope
    axs[0].add_collection(PatchCollection(XZ_hods, 
                                            facecolor="green", 
                                            alpha = .2, 
                                            edgecolor="green"))
    # plot passive
    axs[0].add_collection(PatchCollection([Rectangle((0., volume.get_passives()[-1].z - volume.get_passives()[-1].size), 
                                                    volume.get_passives()[0].lw[0], 
                                                    len(volume.get_passives()) * volume.get_passives()[0].size)], 
                                                    facecolor="blue", 
                                                    alpha = .2, 
                                                    edgecolor="blue"))
    def normalize(x, xmin, xmax):
        return (x - xmin)/(xmax - xmin)
    
    # plot panels
    for p in panels:
        axs[0].axhline(
            y = p["z"], 
            color = "red", 
            xmin = normalize(p["x"] - p["dx"] / 2, 
                            axs[0].get_xlim()[0], 
                            axs[0].get_xlim()[1]), 
            xmax = normalize(p["x"] + p["dx"] / 2, 
                            axs[0].get_xlim()[0], 
                            axs[0].get_xlim()[1])
                        )
        
    axs[0].set_xlabel("x [m]")
    axs[0].set_ylabel("z [m]")
    axs[0].set_title(" YZ view")

    # YZ view
    YZ_hods = [Rectangle((hod["y"], hod["z"]), 
                            hod["dy"], 
                            hod["dz"]) for hod in hods]
    # plot hodoscope
    axs[1].add_collection(PatchCollection(YZ_hods, 
                                            facecolor="green", 
                                            alpha = .2, 
                                            edgecolor="green"))
    # plot passives
    axs[1].add_collection(PatchCollection([Rectangle((0., volume.get_passives()[-1].z - volume.get_passives()[-1].size), 
                                                    volume.get_passives()[0].lw[1], 
                                                    len(volume.get_passives()) * volume.get_passives()[0].size)], 
                                                    facecolor="blue", 
                                                    alpha = .2, 
                                                    edgecolor="blue"))

    # plot panels
    for p in panels:
        axs[1].axhline(
            y = p["z"], 
            color = "red", 
            xmin = normalize(p["y"] - p["dy"] / 2, 
                            axs[1].get_xlim()[0], 
                            axs[1].get_xlim()[1]), 
            xmax = normalize(p["y"] + p["dy"] / 2, 
                            axs[1].get_xlim()[0], 
                            axs[1].get_xlim()[1])
        )

    axs[1].set_xlabel("y [m]")
    axs[1].set_ylabel("z [m]")
    axs[1].set_title(" YZ view")


    # XY view
    # plot hodoscopes
    XY_hods = [Rectangle((hod["x"], hod["y"]), 
                         hod["dx"], 
                         hod["dy"]) for hod in hods]
    
    axs[2].add_collection(PatchCollection(XY_hods, 
                                          facecolor="green", 
                                          alpha = .2, 
                                          edgecolor="green"))

    # plot panels
    XY_planes = [Rectangle((p["x"] - p["dx"] / 2, p["y"]- p["dy"] / 2), 
                            p["dx"], 
                            p["dy"]) for p in panels]
    
    axs[2].add_collection(PatchCollection(XY_planes, 
                                          facecolor="red", 
                                          alpha = .2, 
                                          edgecolor="red"))

    # plot passive
    axs[2].add_collection(PatchCollection([Rectangle((0., 0.), 
                                                    volume.get_passives()[0].lw[0], 
                                                    volume.get_passives()[0].lw[1])], 
                                                    facecolor="blue", 
                                                    alpha = .2, 
                                                    edgecolor="blue"))

    axs[2].set_xlabel("x [m]")
    axs[2].set_ylabel("y [m]")
    axs[2].set_title("XY view")

    if hits is not None:
        axs[0].scatter(hits[event, :, 0].detach().numpy(), hits[event, :, 2].detach().numpy(), marker = "+")
        axs[1].scatter(hits[event, :, 1].detach().numpy(), hits[event, :, 2].detach().numpy(), marker = "+")
        axs[2].scatter(hits[event, :, 0].detach().numpy(), hits[event, :, 1].detach().numpy(), marker = "+")

    if pocas is not None:
        axs[0].scatter(pocas[event, 0].detach().numpy(), pocas[event, 2].detach().numpy(), marker = "+")
        axs[1].scatter(pocas[event, 1].detach().numpy(), pocas[event, 2].detach().numpy(), marker = "+")
        axs[2].scatter(pocas[event, 0].detach().numpy(), pocas[event, 1].detach().numpy(), marker = "+")

    plt.tight_layout()
    plt.show()

def plot_poca_points(pocas: Tensor, binning_xyz: Tuple[float]) -> None:
    r"""
    Plot the poca locations a 2d histograms in the XZ, YZ and XY projection.

    Arguments:
        pocas: The poca points locations (n_muons, 3).
        binning_xyz: The number of bins along x, y, and z (Nx, Ny, Nz).
    """
    fig, axs = plt.subplots(ncols = 3)

    xy_labels = [("x", "z"), ("y", "z"), ("x", "y")]
    unit = " [m]"

    #XZ view
    axs[0].hist2d(pocas[:, 0], pocas[:, 2], bins = (binning_xyz[0], binning_xyz[2]))
    axs[0].set_aspect("equal")

    #YZ view
    axs[1].hist2d(pocas[:, 1], pocas[:, 2], bins = (binning_xyz[1], binning_xyz[2]))
    axs[1].set_aspect("equal")

    #XY view
    axs[2].hist2d(pocas[:, 0], pocas[:, 1], bins = (binning_xyz[0], binning_xyz[1]))
    axs[2].set_aspect("equal")

    for ax, xy_label in zip(axs, xy_labels):
        ax.set_xlabel(xy_label[0] + unit)
        ax.set_xlabel(xy_label[1] + unit)

    plt.tight_layout()
    plt.show()