# QuREBB: Quantum Remote Entanglement Building Block Simulations

Welcome to the **QuREBB** repository! Here, we introduce a modular theoretical framework designed for the comprehensive understanding and comparison of various photon-mediated remote entanglement protocols (REPs). This repository is based on the framework and results outlined in the paper ["Tutorial: Remote entanglement protocols for stationary qubits with photonic interface"](URL_PLACEHOLDER). Structured into four distinct layers, this framework facilitates the assembly of modules by connecting one's output to another's input, elucidating the intricate mechanisms underpinning entanglement.

The primary strength of our framework is its capability to delineate common features inherent to different remote entanglement protocols. Beyond this, it is designed with adaptability at its core, enabling users to modify modules with precision. This adaptability is crucial to compare a protocol across diverse quantum hardware setups or to test an array of protocols on a consistent hardware foundation. Such modularity and precision often remain elusive in dedicated entanglement simulations tailored for specific experimental setups. Dive deep into the intricacies of photon-mediated entanglement generation between qubit systems, and leverage the robust capabilities of our simulation suite.

## Authors

[**Hans Beukers**](mailto:)<sup>1</sup>, [**Matteo Pasini**](mailto:)<sup>1</sup>, [**Hyeongrak Choi**](mailto:)<sup>2</sup>, [**Dirk Englund**](mailto:englund@mit.edu)<sup>2</sup>, [**Ronald Hanson**](mailto:R.Hanson@tudelft.nl)<sup>1</sup>, [**Johannes Borregaard**](mailto:J.Borregaard@tudelft.nl)<sup>1</sup>

<sup>1</sup> QuTech, Delft University of Technology, PO Box 5046, 2600 GA Delft, The Netherlands  
<sup>2</sup> Research Laboratory of Electronics, Massachusetts Institute of Technology, Cambridge, Massachusetts 02139, USA


## Structure 
![GitHub last commit](https://img.shields.io/github/last-commit/QuTech-Delft/QuREBB)

### lib
- **NQobj.py**
	  This file contains the NQobj class, which allows for named indexing of the Qobj of QuTiP
- **quantum_optical_modelling.py**
	  This file contains the quantum optical models that are used to simulate the hardware.
	  The functions in this file get the physical parameters as arguments and return give parameters that describe the response.
	  e.g. The cavity function gets kappa, gamma, g etc. as arguments and returns t, r and l (transmission, reflection and loss).
- **PBB.py**
	  This file contain functions that return the unitary operator describing the quantum channel of a PBB.
	  Some of the functions are stand alone, more complicated PBBs use the quantum_optical_modelling file for simulation of the hardware.
- **LBB.py**
	  Similar in function to the PBB file, but here the functions of the PBB are used to build the LBB quantum channels.
	  These function take in a density matrix and return the updated one.
- **protocol.py**
	  This file contains the class Protocol that is used to simulate the behaviour of a whole protocol.
	  It also provides the class ProtocolSweep for sweeping values in the protocols for fidelity and rate optimization.
- **states.py** 
	  This file is for convenience and its use increases the readability of the code e.g. vacuum() in stead of qt.basis(0,2)

### protocols
- **tutorial_protocols.py**
		The three protocols from the paper (A, B, C) are implemented in code in this file.

### tutorial_simulations
- notebooks
- simulation_data

## Running QuREBB
[![Pipenv](https://img.shields.io/badge/pipenv-locked-brightgreen)](https://pipenv.pypa.io/)

The virtual environment in this QuREBB repository is managed by [pipenv](https://pipenv.pypa.io/en/latest/).
To install pipenv you can use pip:

```bash
$ pip install pipenv --user
```

Using pipenv you then want to install all the dependencies of the repository by syncing with pipenv.
First navigate in the terminal to the QuREBB folder. Then run:

```bash
$ pipenv sync
```

Once you are synced you can go into a shell of this virtual enviroment (again while being in the QuREBB folder):

```bash
$ pipenv shell
```

The commands you now run are within this virtual enviroment, e.g.:

```bash
$ jupyter lab
```

## Dependencies 

The dependencies are formulated in the pipenv file.
Also the pipenv tool can be used to load the correct enviroment (see "running QuREBB").

## License
[![license](https://img.shields.io/badge/license-New%20BSD-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

QuREBB is licensed under the BSD 3-Clause License, which allows for the free use, modification, and distribution of software as long as certain conditions are met.

See the LICENSE.txt file for more details.

## Citing QuREBB

If you use QuREBB in your research, please cite the original [QuREBB paper](URL_PLACEHOLDER).