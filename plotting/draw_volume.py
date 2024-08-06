import matplotlib.pyplot as plt
import numpy as np

from volume.hodoscopelayer import HodoscopeDetectorLayer
from tomopt.volume import Volume
from tomopt.volume.layer import PassiveLayer

from torch import Tensor
from typing import Any, List, Tuple
from tomopt.volume import DetectorHeatMap
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def draw(
    volume: Volume,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    zlim: Tuple[float, float],
) -> None:
    r"""
    Draws the layers/panels pertaining to the volume.
    When using this in a jupyter notebook, use "%matplotlib notebook" to have an interactive plot that you can rotate.

    Arguments:
        xlim: the x axis range for the three-dimensional plot.
        ylim: the y axis range for the three-dimensional plot.
        zlim: the z axis range for the three-dimensional plot.
    """
    ax = plt.figure(figsize=(9, 9)).add_subplot(projection="3d")
    ax.computed_zorder = False
    # TODO: find a way to fix transparency overlap in order to have passive layers in front of bottom active layers.
    passivearrays: List[Any] = []
    activearrays: List[Any] = []

    for layer in volume.layers:
        # fmt: off
            if isinstance(layer, PassiveLayer):
                lw, thez, size = layer.get_lw_z_size()
                roundedz = np.round(thez.item(), 2)
                # TODO: split these to allow for different alpha values (want: more transparent in front, more opaque in the back)
                rect = [
                    [
                        (0, 0, roundedz - size),
                        (0 + lw[0].item(), 0, roundedz - size),
                        (0 + lw[0].item(), 0 + lw[1].item(), roundedz - size),
                        (0, 0 + lw[1].item(), roundedz - size)
                    ],
                    [
                        (0, 0, roundedz - size),
                        (0 + lw[0].item(), 0, roundedz - size),
                        (0 + lw[0].item(), 0, roundedz),
                        (0, 0, roundedz)
                    ],
                    [
                        (0, 0 + lw[1].item(), roundedz - size),
                        (0 + lw[0].item(), 0 + lw[1].item(), roundedz - size),
                        (0 + lw[0].item(), 0 + lw[1].item(), roundedz),
                        (0, 0 + lw[1].item(), roundedz)
                    ],
                    [
                        (0, 0, roundedz - size),
                        (0, 0 + lw[1].item(), roundedz - size),
                        (0, 0 + lw[1].item(), roundedz),
                        (0, 0, roundedz)
                    ],
                    [
                        (0 + lw[0].item(), 0, roundedz - size),
                        (0 + lw[0].item(), 0 + lw[1].item(), roundedz - size),
                        (0 + lw[0].item(), 0 + lw[1].item(), roundedz),
                        (0 + lw[0].item(), 0, roundedz)
                    ],
                ]

                col = "blue"

                passivearrays.append([rect, col, roundedz, 1])
                continue
            # if not passive layer...
            if isinstance(layer, HodoscopeDetectorLayer):
                for h in layer.hodoscopes:
                    lw, thez, size = 2*h.xy, h.z, h.xyz_span[2].item()
                    roundedz = np.round(thez.item(), 2)
                    # TODO: split these to allow for different alpha values (want: more transparent in front, more opaque in the back)

                    rect = [
                        [
                            (h.xy.data[0].item() - h.xyz_span.data[0] / 2.0, h.xy.data[1].item() - h.xyz_span.data[1] / 2.0, h.z.data.item()),
                            (h.xy.data[0].item() + h.xyz_span.data[0] / 2.0, h.xy.data[1].item() - h.xyz_span.data[1] / 2.0, h.z.data.item()),
                            (h.xy.data[0].item() + h.xyz_span.data[0] / 2.0, h.xy.data[1].item() + h.xyz_span.data[1] / 2.0, h.z.data.item()),
                            (h.xy.data[0].item() - h.xyz_span.data[0] / 2.0, h.xy.data[1].item() + h.xyz_span.data[1] / 2.0, h.z.data.item())
                        ],
                        [
                            (h.xy.data[0].item() - h.xyz_span.data[0] / 2.0, h.xy.data[1].item() - h.xyz_span.data[1] / 2.0, h.z.data.item() - size),
                            (h.xy.data[0].item() + h.xyz_span.data[0] / 2.0, h.xy.data[1].item() - h.xyz_span.data[1] / 2.0, h.z.data.item()- size),
                            (h.xy.data[0].item() + h.xyz_span.data[0] / 2.0, h.xy.data[1].item() + h.xyz_span.data[1] / 2.0, h.z.data.item()- size),
                            (h.xy.data[0].item() - h.xyz_span.data[0] / 2.0, h.xy.data[1].item() + h.xyz_span.data[1] / 2.0, h.z.data.item()- size)
                        ],
                        [
                            (h.xy.data[0].item() - h.xyz_span.data[0] / 2.0 , h.xy.data[1].item() - h.xyz_span.data[1] / 2.0, h.z.data.item() - size),
                            (h.xy.data[0].item() - h.xyz_span.data[0] / 2.0 , h.xy.data[1].item() - h.xyz_span.data[1] / 2.0, h.z.data.item()),
                            (h.xy.data[0].item() + h.xyz_span.data[0] / 2.0, h.xy.data[1].item() - h.xyz_span.data[1] / 2.0, h.z.data.item()),
                            (h.xy.data[0].item() + h.xyz_span.data[0] / 2.0, h.xy.data[1].item() - h.xyz_span.data[1] / 2.0, h.z.data.item() - size),
                        ],
                        [
                            (h.xy.data[0].item() - h.xyz_span.data[0] / 2.0 , h.xy.data[1].item() + h.xyz_span.data[1] / 2.0, h.z.data.item() - size),
                            (h.xy.data[0].item() - h.xyz_span.data[0] / 2.0 , h.xy.data[1].item() + h.xyz_span.data[1] / 2.0, h.z.data.item()),
                            (h.xy.data[0].item() + h.xyz_span.data[0] / 2.0, h.xy.data[1].item() + h.xyz_span.data[1] / 2.0, h.z.data.item()),
                            (h.xy.data[0].item() + h.xyz_span.data[0] / 2.0, h.xy.data[1].item() + h.xyz_span.data[1] / 2.0, h.z.data.item() - size),
                        ],
                    
                    
                    ]

                   
                    col = "green"
                    activearrays.append([rect, col, roundedz, 0.1])
                    


                for i, p in layer.yield_zordered_panels():
                    if isinstance(p, DetectorHeatMap):
                        raise TypeError("Drawing not supported yet for DetectorHeatMap panels")
                    col = "red" 
                    if not isinstance(p.xy, Tensor):
                        raise ValueError("Panel xy is not a tensor, for some reason")
                    if not isinstance(p.z, Tensor):
                        raise ValueError("Panel z is not a tensor, for some reason")
                    rect = [[
                        (p.xy.data[0].item() - p.get_scaled_xy_span().data[0] / 2.0, p.xy.data[1].item() - p.get_scaled_xy_span().data[1] / 2.0, p.z.data[0].item()),
                        (p.xy.data[0].item() + p.get_scaled_xy_span().data[0] / 2.0, p.xy.data[1].item() - p.get_scaled_xy_span().data[1] / 2.0, p.z.data[0].item()),
                        (p.xy.data[0].item() + p.get_scaled_xy_span().data[0] / 2.0, p.xy.data[1].item() + p.get_scaled_xy_span().data[1] / 2.0, p.z.data[0].item()),
                        (p.xy.data[0].item() - p.get_scaled_xy_span().data[0] / 2.0, p.xy.data[1].item() + p.get_scaled_xy_span().data[1] / 2.0, p.z.data[0].item())
                    ]]

                    activearrays.append([rect, col, p.z.data[0].item(), 0.2])
            else:
                raise TypeError("Volume.draw does not yet support layers of type", type(layer))

    allarrays = activearrays + passivearrays
    allarrays.sort(key=lambda x: x[2])

    for voxelandcolour in allarrays:
        ax.add_collection3d(
            Poly3DCollection(
                voxelandcolour[0],
                facecolors=voxelandcolour[1],
                linewidths=1,
                edgecolors=voxelandcolour[1],
                alpha=voxelandcolour[3],
                zorder=voxelandcolour[2],
                sort_zpos=voxelandcolour[2],
            )
        )

    plt.ylim(xlim)
    plt.xlim(ylim)
    ax.set_zlim(zlim)
    plt.title("Volume layers")
    red_patch = mpatches.Patch(color="red", label="Active Detector Layers")
    pink_patch = mpatches.Patch(color="blue", label="Passive Layers")
    ax.legend(handles=[red_patch, pink_patch])
    plt.show()
