# TomOptCargo: Differential Muon Tomography Optimisation for Border Control Applications

This repo provides a plug-in library for the differential optimisation of scattering muon tomography systems applied to border control applications. For a quick overview, check out our paper submitted to the Maresec conference [PAPER LINK WHEN AVAILABLE]() and, for a more in-depth guide to the core TomOpt software, please read our first publication [Giles C Strong et al 2024 Mach. Learn.: Sci. Technol. 5 035002](10.1088/2632-2153/ad52e7).

As a disclaimer, this is a library designed to be extended by users for their specific tasks: e.g. passive volume definition, inference methods, and loss functions. Additionally, optimisation in TomOptCargo can be unstable, and requires careful tuning by users. This is to say that it is not a polished product for the general public, but rather fellow researchers in the field of optimisation and muon tomography.

If you are interested in using this library seriously, please contact us;  we would love to here if you have a specific use-case you wish to work on.


## Overview

The TomOpt library is designed to optimise the design of a muon tomography system. The TomOptCargo library extends the capabilities of TomOpt to include hodoscopes, structures that can hold in a fixed position a number of detector panels: the hodoscope in turn is let free to move in the optimization, leading to a different kind of optimization constraints with respect to the core TomOpt. As in TomOpt, the detector system is defined by a set of parameters, which are used to define the geometry of the detectors. The optimisation is performed by minimising a loss function, which is defined by the user. The loss function is evaluated by simulating the muon scattering process through the detector system and passive volumes. The information recorded by the detectors is then passed through an inference system to arrive at a set of task-specific parameters. These are then compared to the ground truth, and the loss is calculated. The gradient of the loss with respect to the detector parameters is then used to update the detector parameters.

The TomOptCargo library is designed to be modular, and to allow for the easy addition of new inference systems, loss functions, and passive volume definitions. The library is also designed to be easily extensible to new optimisation algorithms, and to allow for the easy addition of new constraints on the detector parameters.

TomOptCargo consists of the implementation of several submodules:

- volume: contains classes for defining passive volumes and detector systems in form of hodoscopes.
- optimisation: provides classes for handling the optimisation of detector parameters in the case of hodoscopes.
- plotting: various plotting utilities for visualising the detector system, the optimisation process, and results

## Installation

### Dependencies

You should install `tomopt` and all its related dependencies via e.g.

```bash
pip install tomopt
```

Then, you should clone this repository, and you can run the enclosed notebooks.

```bash
git clone git@github.com:vischia/TomOptCargo.git
cd TomOptCargo
```
    
### For development

Follow the instructions in the [tomopt](github.com:GilesStrong/tomopt/) README file for the installation for developers, then install `TomOptCargo` as outlined above.

## Examples

A few examples are included to introduce users and developers to the TomOptCargo library. These take the form of Jupyter notebooks. In `examples/` there are four ordered notebooks:

- `tomopt_cargo.ipynb` aims to show the user the basic usage of TomOptCargo  and the general workflow.
- `scatter_batch_filter_muons_dev.ipynb`, `overlap_z_no_hits_error_dev.ipynb`, and `muon_prop_through_hodoscopes.ipynb` aim to illustrate specific details of the optimization in case of hodoscopes.
- `hodoscopeTest.ipynb` aims at illustrating the core classes of TomOptCargo.
- `draw_volume_2D_dev.ipynb` aims to illustrate the visualization capabilities of the library.

The user is also encourage to check the `examples/` folder of the core TomOpt.


### Running notebooks in a remote cluster

If you want to run notebooks on a remote cluster but access them on the browser of your local machine, you need to forward the notebook server from the cluster to your local machine.

On the cluster, run:
```
poetry run jupyter notebook --no-browser --port=8889
```

On your local computer, you need to set up a forwarding that picks the flux of data from the cluster via a local port, and makes it available on another port as if the server was in the local machine:
```
ssh -N -f -L localhost:8888:localhost:8889 username@cluster_hostname
```

The layperson version of this command is: *take the flux of info from the port `8889` of `cluster_hostname`, logging in as `username`, get it inside the local machine via the port `8889`, and make it available on the port `8888` as if the jupyter notebook server was running locally on the port `8888`*

You can now point your browser to [http://localhost:8888/tree](http://localhost:8888/tree) (you will be asked to copy the server authentication token, which is the number that is shown by jupyter when you run the notebook on the server)

If there is an intermediate machine (e.g. a gateway) between the cluster and your local machine, you need to set up a similar port forwarding on the gateway machine. The crucial point is that the input port of each machine must be the output port of the machine before it in the chain. For instance:
```
jupyter notebook --no-browser --port=8889 # on the cluster
ssh -N -f -L localhost:8888:localhost:8889 username@cluster_hostname # on the gateway. Makes the notebook running on the cluster port 8889 available on the local port 8888
ssh -N -f -L localhost:8890:localhost:8888 username@gateway_hostname # on your local machine. Picks up the server available on 8888 of the gateway and makes it available on the local port 8890 (or any other number, e.g. 8888)
```

## External repos

N.B. Most are not currently public

- [tomopt](https://github.com/GilesStrong/tomopt) public. Contains the core functionality

## Authors

The TomOptCargo project, and its continued development and support, is the result of the combined work of many people, whose contributions are summarised in [the author list](https://github.com/vischia/TomOptCargo/blob/main/AUTHORS.md)
