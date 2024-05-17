from tomopt.core import *
from typing import List, Tuple, Union

import torch
from torch import Tensor
import numpy as np
import random

from tomopt.volume import PassiveLayer

uranium = X0['uranium']
lead = X0['lead']
del X0['uranium'] ; del X0['lead']

class RandomPassive:
    r"""
    Generates a passive layer and material configuration. 
    By default the methods return the material configuration and not the constructed volume,
    this can be changed setting gen_layers to True

    There are three preconfigured passive options:
        - simple: <1 m^3 volume of a random material with a block of another random material.
        - medium: cube of 1-2 m sides with 3 random material blocks inside.
        - complex: truck-like passive with steel frame and dimensions 6 x 2.5 x 2.5 m^3.

    A custom volume can be made calling the .custom() method.

    Arguments:
        lwh: dimensions of the passive
        voxel_size
        n_blocks: number of random material blocks
        heavy_material_chance: if not None, a block of a heavy material (lead/uranium) might appear
        truck_frame: .complex() and .custom() only. If True, the generator has a truck-like material distrubution around the borders
        gen_layers: if True, returns a list of passive layers configured based on lwh and voxel_size      
    """

    def __init__(self,
                 lwh: Tensor = None,
                 n_blocks: int = None,
                 heavy_material_chance: float = .5,
                 truck_frame: bool = True,
                 get_layers: bool = False,
                 device: torch.device = DEVICE,
                 ) -> None:
        
        self.lwh = lwh
        self.n_blocks = n_blocks
        self.heavy_material_chance = heavy_material_chance
        self.truck_frame = truck_frame
        self.get_layers = get_layers
        self.device = device

        self.voxel_size = None
        self.should_open = False
        
    def simple(self) -> Union[np.ndarray, Tuple[List[PassiveLayer], np.ndarray]]:

        self.lwh = Tensor([0.1*np.random.randint(4, 9)]*3)
        self.voxel_size = 0.1
        self.rmidx = [np.random.randint(0, len(X0)), np.random.randint(0, len(X0))]
        self.rm = [list(X0.values())[i] for i in self.rmidx]

        if self.heavy_material_chance != False:
            if random.choice(np.arange(0, 1, self.heavy_material_chance)) == False:
                self.rm[-1] = random.choice([uranium, lead])
                self.should_open = True

        block_low, block_high = self.get_block_coords()

        def generator(*, 
                      z: Tensor, 
                      lw: Tensor, 
                      size: float,
                      rm: List[float] = self.rm
                      ) -> Tensor:
        
            rad_length = torch.ones(list((lw / size).long()))*rm[0]

            low_xy = np.round(block_low[:2] / size).astype(int)
            high_xy = np.round(block_high[:2] / size).astype(int)
            if z >= block_low[2] and z <= block_high[2]:
                rad_length[low_xy[0] : high_xy[0], low_xy[1] : high_xy[1]] = rm[1]

            return rad_length
        
        if not self.get_layers:
            return generator, self.should_open
        else:
            return self.gen_passive_layers(), generator, self.should_open

    def medium(self) -> Union[np.ndarray, Tuple[List[PassiveLayer], np.ndarray]]:

        self.lwh = Tensor([0.1*np.random.randint(12, 20)]*3)
        self.voxel_size = float(self.lwh[-1]/8)
        if self.n_blocks is None:
            self.n_blocks = 4
        self.rmidx = [(np.random.randint(0, len(X0))) for i in range(self.n_blocks)]
        self.rm = [list(X0.values())[i] for i in self.rmidx]

        if self.heavy_material_chance != False:
            if random.choice(np.arange(0, 1, self.heavy_material_chance)) == False:
                self.rm[-1] = random.choice([uranium, lead])
                self.should_open = True

        block_low, block_high = [], []
        
        for i in self.rm:
            low, high = self.get_block_coords()
            block_low.append(low)
            block_high.append(high)

        def generator(*,
                      z: Tensor, 
                      lw: Tensor, 
                      size: float,
                      rm: List[float] = self.rm
                      ) -> Tensor:
            
            rad_length = torch.ones(list((lw / size).long()))*X0['water']

            for i in range(len(rm)):
                low_xy = np.round(block_low[i][:2] / size).astype(int)
                high_xy = np.round(block_high[i][:2] / size).astype(int)
                if z >= block_low[i][2] and z <= block_high[i][2]:
                    rad_length[low_xy[0] : high_xy[0], low_xy[1] : high_xy[1]] = rm[i]
            
            return rad_length

        if not self.get_layers:
            return generator, self.should_open
        else:
            return self.gen_passive_layers(), generator, self.should_open

    def complex(self) -> Union[np.ndarray, Tuple[List[PassiveLayer], np.ndarray]]:
        
        self.lwh = Tensor([6, 2.5, 2.5])
        self.voxel_size = float(self.lwh[-1]/10)
        if self.n_blocks is None:
            self.n_blocks = 15
        self.rmidx = [(np.random.randint(0, len(X0))) for i in range(self.n_blocks)]
        self.rm = [list(X0.values())[i] for i in self.rmidx]

        if self.heavy_material_chance != False:
            if random.choice(np.arange(0, 1, self.heavy_material_chance)) == False:
                self.rm[-1] = random.choice([uranium, lead])
                self.should_open = True

        block_low, block_high = [], []
        
        for i in self.rm:
            low, high = self.get_block_coords()
            block_low.append(low)
            block_high.append(high)

        def generator(*,  
                      z: Tensor,
                      lw: Tensor, 
                      size: float, 
                      rm: List[float] = self.rm, 
                      z_min: float = self.voxel_size,
                      z_max: float = self.lwh[-1],
                      truck_frame: bool = self.truck_frame
                      ) -> Tensor:
            
            rad_length = torch.ones(list((lw / size).long()))*X0['water']

            for i in range(len(rm)):
                low_xy = np.round(block_low[i][:2] / size).astype(int)
                high_xy = np.round(block_high[i][:2] / size).astype(int)
                if z >= block_low[i][2] and z <= block_high[i][2]:
                    rad_length[low_xy[0] : high_xy[0], low_xy[1] : high_xy[1]] = rm[i]
            
            if truck_frame:
                rad_length[:,0] = X0['aluminium']
                rad_length[:,-1] = X0['aluminium']
                rad_length[0,:] = X0['aluminium']
                rad_length[-1,:] = X0['aluminium']
                if z == z_max: rad_length[:,:]= X0['aluminium']
                if z == z_min: rad_length[:,:]= X0['steel']

            return rad_length

        if not self.get_layers:
            return generator, self.should_open
        else:
            return self.gen_passive_layers(), generator, self.should_open
        
    def custom(self) -> Union[np.ndarray, Tuple[List[PassiveLayer], np.ndarray]]:
        
        self.voxel_size = float(self.lwh[-1]/10)
        self.rmidx = [(np.random.randint(0, len(X0))) for i in range(self.n_blocks)]
        self.rm = [list(X0.values())[i] for i in self.rmidx]

        if self.heavy_material_chance != False:
            if random.choice(np.arange(0, 1, self.heavy_material_chance)) == False:
                self.rmidx[-1] = random.choice([uranium, lead])
                self.should_open = True

        block_low, block_high = [], []
        
        for i in self.rm:
            low, high = self.get_block_coords()
            block_low.append(low)
            block_high.append(high)

        def generator(*,  
                      z: Tensor,
                      lw: Tensor, 
                      size: float, 
                      rm = self.rm, 
                      z_min: float = self.voxel_size,
                      z_max: float = self.lwh[-1],
                      truck_frame = self.truck_frame) -> Tensor:
            
            rad_length = torch.ones(list((lw / size).long()))*X0['water']

            for i in range(len(rm)):
                low_xy = np.round(block_low[i][:2] / size).astype(int)
                high_xy = np.round(block_high[i][:2] / size).astype(int)
                if z >= block_low[i][2] and z <= block_high[i][2]:
                    rad_length[low_xy[0] : high_xy[0], low_xy[1] : high_xy[1]] = rm[i]
            
            if truck_frame:
                rad_length[:,0] = X0['aluminium']
                rad_length[:,-1] = X0['aluminium']
                rad_length[0,:] = X0['aluminium']
                rad_length[-1,:] = X0['aluminium']
                if z == z_max: rad_length[:,:]= X0['aluminium']
                if z == z_min: rad_length[:,:]= X0['steel']

            return rad_length

        if not self.get_layers:
            return generator, self.should_open
        else:
            return self.gen_passive_layers(), generator, self.should_open

    def get_block_coords(self) -> Tuple[np.ndarray, np.ndarray]:
        self.block_size_max = self.lwh/2
    
        block_size = np.hstack(
            (
                np.random.uniform(self.voxel_size*2, self.block_size_max[0]),
                np.random.uniform(self.voxel_size*2, self.block_size_max[1]),
                np.random.uniform(self.voxel_size*2, self.block_size_max[2]),
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
        for z in np.arange(self.lwh[-1], 0, -self.voxel_size):
            layers.append(PassiveLayer(lw=self.lwh[:2], z=z, size=self.voxel_size, device=self.device))
        
        return layers
