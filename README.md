# QuREBB

Quantum Remote Entanglement Building Block simultions

This repository facilitates simulation of remote entanglement protocols.
It is based on the framework outlined in the paper
Tutorial: Remote entanglement protocols for stationary qubits with photonic interface

# Structure 

The structure is as follows:
- lib
	- NQobj.py - 
	  This file contains the NQobj class, which allows for named indexing of the Qobj of QuTiP
	- quantum_optical_modelling.py - 
	  This file contains the quantum optical models that are used to simulate the hardware.
	  The functions in this file get the physical parameters as arguments and return give parameters that describe the response.
	  e.g. The cavity function gets kappa, gamma, g etc. as arguments and returns t, r and l (transmission, reflection and loss).
	- PBB.py - 
	  This file contain functions that return the unitary operator describing the quantum channel of a PBB.
	  Some of the functions are stand alone, more complicated PBBs use the quantum_optical_modelling file for simulation of the hardware.
	- LBB.py - 
	  Similar in function to the PBB file, but here the functions of the PBB are used to build the LBB quantum channels.
	  These function take in a density matrix and return the updated one.
	- protocol.py - 
	  This file contains the class Protocol that is used to simulate the behaviour of a whole protocol.
	  It also provides the class ProtocolSweep for sweeping values in the protocols for fidelity and rate optimization.
	- states.py - 
	  This file is for convenience and its use increases the readability of the code e.g. vacuum() in stead of qt.basis(0,2)

- protocols
	- tutorial_protocols.py
		The three protocols from the paper (A, B, C) are implemented in code in this file.

# Dependencies 

The dependencies are formulated in the pipenv file.
Also the pipenv tool can be used to load the correct enviroment.
