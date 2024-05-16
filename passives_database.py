from tomopt.core import *
from typing import List, Tuple, Union

import torch
from torch import Tensor
import numpy as np

from tomopt.volume import PassiveLayer


class RandomPassive:
    r'''
    Generates a passive layer and material configuration. 
    
    There are three preconfigured passive options:
        - simple: <1 m^3 volume of a random material with a block of another random material.
        - medium: cube of 1-2 m sides with 3 random material blocks inside.
        - complex: truck-like passive with steel frame and dimensions 6 x 2.5 x 2.5 m^3.

    A custom volume can be made calling the .custom() method.

    Arguments:

        
        
    '''

    def __init__(self,
                 lwh: Tensor = None,
                 voxel_size: float = None,
                 n_blocks: int = None,
                 heavy_material: bool = None,
                 gen_layers: bool = False,
                 device: torch.device = DEVICE,
                 ) -> None:
        
        self.lwh = lwh
        self.voxel_size = voxel_size
        self.n_blocks = n_blocks
        self.heavy_material = heavy_material
        self.gen_layers = gen_layers
        self.device = device
        
    def simple(self) -> Union[np.ndarray, Tuple[List[PassiveLayer], np.ndarray]]:

        self.lwh = Tensor([0.1*np.random.randint(4, 9)]*3)
        self.voxel_size = 0.1
        self.block_size_max = self.lwh/2
        self.rmidx = [np.random.randint(0, len(X0)), np.random.randint(0, len(X0))]

        block_low, block_high = self.get_block_coords()

        def generator(*, rmidx = self.rmidx, z: Tensor, lw: Tensor, size: float) -> Tensor:
            low_xy = np.round(block_low[:2] / size).astype(int)
            high_xy = np.round(block_high[:2] / size).astype(int)
            rad_length = torch.ones(list((lw / size).long()))*list(X0.values())[rmidx[0]]
            if z >= block_low[2] and z <= block_high[2]:
                rad_length[low_xy[0] : high_xy[0], low_xy[1] : high_xy[1]] = list(X0.values())[rmidx[1]]
            return rad_length

        if self.gen_layers is False:
            return generator
        else:
            return self.gen_passive_layers(), generator

    def medium(self) -> Union[np.ndarray, Tuple[List[PassiveLayer], np.ndarray]]:

        self.lwh = Tensor([np.random.randint(1, 2)]*3)
        self.voxel_size = 0.2
        if self.n_blocks is None:
            self.n_blocks = 3
        self.rmidx = [(np.random.randint(0, len(X0))) for i in range(self.n_blocks+1)]

        block_low, block_high = self.get_block_coords()

        def generator(*, rmidx = self.rmidx, z: Tensor, lw: Tensor, size: float) -> Tensor:
            low_xy = np.round(block_low[:2] / size).astype(int)
            high_xy = np.round(block_high[:2] / size).astype(int)
            rad_length = torch.ones(list((lw / size).long()))*list(X0.values())[rmidx[-1]]

            for i in range(len(rmidx)-1):
                if z >= block_low[i] and z <= block_high[i]:
                    rad_length[low_xy[0] : high_xy[0], low_xy[1] : high_xy[1]] = list(X0.values())[rmidx[1]]
            
            return rad_length
        
        if self.gen_layers is False:
            return generator
        else:
            return self.gen_passive_layers(), generator

    # def complex(self) -> Union[np.ndarray, Tuple[List[PassiveLayer], np.ndarray]]:

    def get_block_coords(self) -> Tuple[np.ndarray, np.ndarray]:
    
        block_size = np.hstack(
            (
                np.random.uniform(self.voxel_size, self.block_size_max[0]),
                np.random.uniform(self.voxel_size, self.block_size_max[1]),
                np.random.uniform(self.voxel_size, self.block_size_max[2]),
            )
        )

        block_low = np.hstack(
            (
                np.random.uniform(high=self.lwh[0] - block_size[0]),
                np.random.uniform(high=self.lwh[1] - block_size[1]),
                np.random.uniform(0, self.lwh[-1] - block_size[2]),
            )
        )
        block_high = block_low + block_size

        return block_low, block_high    

    def gen_passive_layers(self) -> List[PassiveLayer]:

        layers = []
        for z in np.arange(self.lwh[-1].item(), 0, step=-self.voxel_size):
            layers.append(PassiveLayer(lw=self.lwh[:2], z=z, size=self.voxel_size, device=self.device))
        
        return layers
