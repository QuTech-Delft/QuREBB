from lib.states import *
import lib.NQobj as nq
import lib.quantum_optical_modelling as qom
import lib.PBB as pbb
import lib.LBB as lbb
from lib.protocol import Protocol, SinglePhotonTwoModes, TwoPhotonsTwoModes

import numpy as np
import qutip as qt
import scipy.constants as cst
import matplotlib.pyplot as plt


class LinearReflection(SinglePhotonTwoModes):

	def __init__(self, name:str, parameters: dict, dim: int = 3, *args, **kwargs):

		super().__init__(name, parameters, dim=dim, *args, **kwargs)

	def _default_herald_state(self) -> nq.NQobj:

		photon_qeyes = []
		vacuum_projs = []
		photon_qeyes.append(nq.name(qt.qeye(self.dim), self.photon_names[0], 'oper'))
		vacuum_projs.append(self.photon_modes[0].proj())

		return nq.tensor(*photon_qeyes) - nq.tensor(*vacuum_projs)

	def _default_target_state(self) -> nq.NQobj:

		return (nq.tensor( nq.name(up, 'Alice', 'state'), nq.name(up , 'Bob', 'state')) +\
				nq.tensor( nq.name(down, 'Alice', 'state'), nq.name(down, 'Bob', 'state') )).unit()


	def run(self):
		"""
		Runs the protocol
		"""

		self.prepare()
		self.dm = self.dm_init

		for pn in self.photon_names:
			self.do_lbb(lbb.ModeLoss, pn, par_name='insertion_loss')
		self.do_lbb(lbb.CavityTimeBinSPI, self.photon_names, 'Alice')
		for pn in self.photon_names:
			self.do_lbb(lbb.ModeLoss, pn, par_name='insertion_loss')

		for pn in self.photon_names:
			self.do_lbb(lbb.ModeLoss, pn)

		for pn in self.photon_names:
			self.do_lbb(lbb.ModeLoss, pn, par_name='insertion_loss')
		self.do_lbb(lbb.CavityTimeBinSPI, self.photon_names, 'Bob')
		for pn in self.photon_names:
			self.do_lbb(lbb.ModeLoss, pn, par_name='insertion_loss')

		self.do_lbb(lbb.BasisRotation, self.photon_names)

		for pn in self.photon_names:
			self.do_lbb(lbb.DarkCounts, pn)

		self.do_measurement(lbb.MeasureProj, self.herald_state)

		return self.metrics()


	def run_spe(self):
		"""
		Runs only Spin-Photon Entanglement
		"""

		spe_herald_state = (nq.tensor(nq.name(up, 'Alice'), nq.name(photon, 'E'), nq.name(vacuum, 'L')) +\
            nq.tensor(nq.name(down, 'Alice'), nq.name(vacuum, 'E'), nq.name(photon, 'L'))).unit()

		self.prepare()

		self.dm = self.dm_init.ptrace(['Bob'], keep=False)

		for pn in self.photon_names:
			self.do_lbb(lbb.ModeLoss, pn, par_name='insertion_loss')
		self.do_lbb(lbb.CavityTimeBinSPI, self.photon_names, 'Alice')
		self.do_lbb(lbb.SpinPiX, spin_name='Alice')
		for pn in self.photon_names:
			self.do_lbb(lbb.ModeLoss, pn, par_name='insertion_loss')

		self.do_measurement(lbb.PhotonDetectionProj, self.photon_spd_projectors, 'Alice', ['E','L'] )

		return self.metrics(target_state = spe_herald_state)

class MidpointReflection(TwoPhotonsTwoModes):

	def __init__(self, name:str, parameters: dict, dim: int = 3, *args, **kwargs):

		super().__init__(name, parameters, dim=dim, *args, **kwargs)

	def _default_herald_state(self) -> nq.NQobj:
		photon_qeyes = []
		vacuum_projs = []

		for name in self.photon_names:
			photon_qeyes.append(nq.name(qt.qeye(self.dim), name, 'oper'))

		for mode in self.photon_modes:
			vacuum_projs.append(mode.proj())
			
		return nq.tensor(
					nq.tensor(*photon_qeyes[0::3]) - nq.tensor(*vacuum_projs[0::3]), 
					nq.tensor(*photon_qeyes[1:-1]) - nq.tensor(*vacuum_projs[1:-1])
				)

	def _default_target_state(self) -> nq.NQobj:

		return (nq.tensor( nq.name(up, 'Alice', 'state'), nq.name(down , 'Bob', 'state')) +\
				nq.tensor( nq.name(down, 'Alice', 'state'), nq.name(up, 'Bob', 'state') )).unit()

	def run(self):

		self.prepare()
		self.dm = self.dm_init

		for pn in self.photon_names[:2]:
			self.do_lbb(lbb.ModeLoss, pn, par_name='insertion_loss')

		self.do_lbb(lbb.CavityTimeBinSPI, self.photon_names[:2], 'Alice')

		for pn in self.photon_names[:2]:
			self.do_lbb(lbb.ModeLoss, pn, par_name='insertion_loss')

		for pn in self.photon_names[2:]:
			self.do_lbb(lbb.ModeLoss, pn, par_name='insertion_loss')

		self.do_lbb(lbb.CavityTimeBinSPI, self.photon_names[2:], 'Bob')

		for pn in self.photon_names[2:]:
			self.do_lbb(lbb.ModeLoss, pn, par_name='insertion_loss')

		for pn in self.photon_names:
			self.do_lbb(lbb.ModeLoss, pn)

		self.do_lbb(lbb.HOM, self.photon_names[0::2])
		self.do_lbb(lbb.HOM, self.photon_names[1::2])

		for pn in self.photon_names:
			self.do_lbb(lbb.DarkCounts, pn)

		self.do_measurement(lbb.MeasureProj, self.herald_state)
		return self.metrics()
		