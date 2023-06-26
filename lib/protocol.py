from lib.states import *
import lib.NQobj as nq
import lib.quantum_optical_modelling as qom
import lib.PBB as pbb
import lib.LBB as lbb


import numpy as np
import qutip as qt
import scipy.constants as cst
import matplotlib.pyplot as plt

import multiprocessing as multi
import functools, itertools
import time
from tqdm.notebook import tqdm


class Protocol:
	"""
	this class handles a two-qubit entanaglement protocol,
	holds the protocol parameters, photon encoding and density matrix

	Attributes:
		name: (str) the protocol name.
		parmeters: (dict) dictionary containing the parameters necessary for protocol operation.
		dim: (int) dimension of the photon space. Default is 3 as this is the minimum for using single photons and HOM interference.
	
	Additional arguments:
		photon_names: (list) list of names for the photonic modes. NB: should this be a mandatory attribute?
		start_state: (NQobj) start state of the protocol
		target_state: (NQobj) the target final state of the protocol (used to calculate fidelity)

	"""

	def __init__(self, name:str, parameters:dict, dim:int =3, **kwargs):

		self.parameters : dict = parameters
		self.sweep_params : dict = kwargs.pop('sweep_params', None)

		self.name: str = name
		self.dm : nq.NQobj = None
		self.dm_init : nq.NQobj = None

		self.fidelity_opt = None
		self.rate_opt = None

		# default nodes are Alice and Bob
		self.alice = nq.name(qt.qeye(2), 'Alice', kind='state')
		self.bob = nq.name(qt.qeye(2), 'Bob', kind='state')

		self.photon_names: [str] = kwargs.pop('photon_names', [])
		self.photon_modes: [nq.NQobj] = []
		self.photon_spd_projectors: [nq.NQobj] = []

		self.dim: int = parameters.pop('dim', dim)
		self.target_state: nq.NQobj = None
		self.herald_state: nq.NQobj = None

		self._perpare_photons()
		self._prepare_blank_dm()
		self._prepare_photon_spd_projectors()
		self.start_state = kwargs.pop('start_state', self._default_start_state())
		self.herald_state = kwargs.pop('herald_state', self._default_herald_state())
		self.target_state = kwargs.pop('target_state', self._default_target_state())

		self.save = kwargs.pop('save',False)
		self.save_folder = './'
		self.save_path = self.save_folder + self.name + '.h5'
		if self.save:
			self._create_data_file()



	def _perpare_photons(self) -> None:
		"""
		Prepares vacuum photon modes for each photon name present
		"""

		for name in self.photon_names:
			self.photon_modes.append(
						nq.name(vacuum_dim(self.dim), name, kind='state')
					)

	def _prepare_photon_spd_projectors(self) -> None:
		"""
		Prepares projectors for the single photon detection with non photon-resolving detectors (projects on everything but vacuum)
		"""
		for name in self.photon_names:
			self.photon_spd_projectors.append( 
						nq.name(qt.qeye(self.dim)-vacuum_dim(self.dim).proj(), name, 'oper') 
						)

	def _prepare_blank_dm(self) -> None:
		"""
		prepares a "blank" (identity) density matrix with the right mode names and dimensions
		"""
		#photon_dm = [nq.ket2dm(x) for x in self.photon_modes]
		photon_dm = [nq.name(qt.qeye(self.dim), x, 'state') for x in self.photon_names]
		self.dm = nq.tensor( self.alice, self.bob, *photon_dm)

	def _default_start_state(self) -> nq.NQobj:
		""" 
		Sets the default start state. Must be defined by subclasses according to the specific protocol 
		"""

		return nq.tensor( nq.name(x, 'Alice', 'state'), nq.name(x, 'Bob', 'state'), *self.photon_modes).unit()

	def _default_herald_state(self) -> nq.NQobj:
		""" 
		Sets the default herald state. Must be defined by subclasses according to the specific protocol 
		"""
		photon_qeyes = []
		vacuum_projs = []
		for n in self.photon_names:
			photon_qeyes.append(nq.name(qt.qeye(self.dim), n, 'oper'))

		for v in self.photon_modes:
			vacuum_projs.append(v.proj())
		return nq.tensor(*photon_qeyes) - nq.tensor(*vacuum_projs)

	def _default_target_state(self) -> nq.NQobj:
		""" 
		Sets the default target state. Must be defined by subclasses according to the specific protocol 
		"""

		return (nq.tensor( nq.name(up, 'Alice', 'state'), nq.name(down , 'Bob', 'state')) +\
				nq.tensor( nq.name(down, 'Alice', 'state'), nq.name(up, 'Bob', 'state') )).unit()

	def run(self) -> (float, float):
		""" Runs the protocol sequence.
			returns:
				fidelity (float)
				rate (float)

		"""
		pass

	def run_with_params(self, params):
		""" Runs the protocol sequence with different parameters than the own 
			returns:
				fidelity (float)
				rate (float)
		"""
		save_params = self.parameters
		self.parameters = params
		f,r = self.run()
		self.paramters = save_params
		return f,r

	def do_lbb(self, LBB,  photon_names=None, spin_name=None, parameters = None, **kwargs):
		""" Makes a logical building block act on the densituy matrix. Uses default parameters by default"""
		if parameters is None:
			parameters = self.parameters

		dm_out = LBB(self.dm, spin_name=spin_name, photon_names=photon_names, parameters=parameters, dim=self.dim, **kwargs)
		self.dm=dm_out

	def do_measurement(self, measurement_lbb, measurement_state, *args, **kwargs):
		""" Makes a measurement LBB act on the density matrix """
		dm_out = measurement_lbb(self.dm, measurement_state, *args, **kwargs)
		self.dm=dm_out


	def prepare(self):
		"""
		Prepares the initial density matrix by projecting the density matrix to the start state
		"""
		self._prepare_blank_dm()
		self.dm_init = self.start_state.proj()*self.dm*self.start_state.proj()

	def metrics(self, **kwargs):
		"""
		Calculates fidelity and success probability versus target spin state
		"""
		use_state = kwargs.get('target_state', self.target_state)
		rate = self.dm.tr()
		fid = nq.fidelity(self.dm.unit(), nq.ket2dm(use_state))**2
		return fid, rate		

	def get_param(self, name:str):
		try:
			return self.parameters[name]
		except:
			print('No parameter named {}'.format(name))
			return None

	def set_param(self, name:str, value):
		try: 
			self.parameters[name] = value
		except:
			print('Operation failed. Maybe parameter {} does not exists.'.format(name))


	""" 
	Utilities for saving, sweeping and optimizing
	"""
	def optimize(self):
			pass

	def update_pars_and_run(self, par_names, *args):
		for i,el in enumerate(par_names):
			self.parameters[el] = args[i]
		return self.run()

	def update_pars_and_run_spe(self, par_names, *args):
		for i,el in enumerate(par_names):
			self.parameters[el] = args[i]
		return self.run_spe()

	def update_pars_and_sweep(self, par_names, *args):
		for i,el in enumerate(par_names):
			self.parameters[el] = args[i]
		return self.multiprocess_sweep()

	def multiprocess_sweep(self, chunksize=1, spe_only = False):
		par_names = list(self.sweep_params.keys())


		#wrap = lambda sweep_list : update_pars_and_run(protocol, params,par_names,*sweep_list)
		wrap = functools.partial(self.update_pars_and_run, par_names)
		if spe_only:
			wrap = functools.partial(self.update_pars_and_run_spe, par_names)



		pars = [self.sweep_params[x]['range'] for x in par_names]
		par_dimensions = [len(x) for x in pars]
		# grid = np.array(np.meshgrid(*pars))
		# XY = np.array([*[x.flatten() for x in grid]]).T
		# args=list(XY)

		ncpu = multi.cpu_count()
		t0=time.time()
		with multi.Pool(ncpu) as processing_pool:
			# accumulate results in a dictionary
			#results = processing_pool.starmap(wrap, args , chunksize=chunksize)
			results = processing_pool.starmap(wrap, itertools.product(*pars))
		t_sim=time.time() - t0
		print('Sweep time with multi was {:.3f} s'.format(t_sim))

	    
		ff = np.array([x[0] for x in results]).reshape(*par_dimensions).T
		rr = np.array([x[1] for x in results]).reshape(*par_dimensions).T

		self.fidelity_opt = ff
		self.rate_opt = rr

		return ff,rr


	def save_data(self):
		pass

	def _create_data_file(self):
		pass

class SinglePhotonTwoModes(Protocol):
	"""
	Protocols with a single photon ancoded in two modes (e.g. polarization or time bin)
	"""

	def __init__(self, name:str, parameters: dict, dim:int=3, *args, **kwargs):

		super().__init__(name, parameters, dim=dim, *args, **kwargs)

		
	def _default_start_state(self):
		"""
		Start state has Alice and Bob in x, and photon in (E+L)
		"""
		if self.parameters['alpha'] is None:
			creators = (nq.name(create_dim(self.dim), self.photon_names[0], 'oper') +\
						 nq.name(create_dim(self.dim), self.photon_names[1], 'oper'))
		else:
			creators = (nq.name(displace_dim(self.dim, self.parameters['alpha']), self.photon_names[0], 'oper') +\
						 nq.name(displace_dim(self.dim, self.parameters['alpha']), self.photon_names[1], 'oper'))
		
		state_oper = nq.tensor( nq.name(qt.qeye(2), 'Alice', 'oper'), nq.name(qt.qeye(2), 'Bob', 'oper'), creators)
		return (state_oper * super()._default_start_state()).unit()


	# def prepare(self):
	# 	self.dm_init = self.start_state.proj()*self.dm*self.start_state.proj()

class TwoPhotonsTwoModes(Protocol):

	"""
	Protocols with two photons encoded in two modes each (e.g. reflection with midpoint detector and two sources)
	"""

	def __init__(self, name:str, parameters: dict, dim:int=3, *args, **kwargs):

		super().__init__(name, parameters, dim=dim, *args, **kwargs)

	def _default_start_state(self):
		"""
		Start state has Alice and Bob in x, and photon in (E+L)
		"""
		if self.parameters['alpha'] is None:
			creators = nq.tensor( 
						(nq.name(create_dim(self.dim), self.photon_names[0], 'oper') +\
						 nq.name(create_dim(self.dim), self.photon_names[1], 'oper')),
						 (nq.name(create_dim(self.dim), self.photon_names[2], 'oper') +\
						 nq.name(create_dim(self.dim), self.photon_names[3], 'oper'))
						 )
		else:
			creators = nq.tensor( 
						(nq.name(displace_dim(self.dim, self.parameters['alpha']), self.photon_names[0], 'oper') +\
						 nq.name(displace_dim(self.dim, self.parameters['alpha']), self.photon_names[1], 'oper')),
						 (nq.name(displace_dim(self.dim, self.parameters['alpha']), self.photon_names[2], 'oper') +\
						 nq.name(displace_dim(self.dim, self.parameters['alpha']), self.photon_names[3], 'oper'))
						 )
		
		state_oper = nq.tensor( nq.name(qt.qeye(2), 'Alice', 'oper'), nq.name(qt.qeye(2), 'Bob', 'oper'), creators)
		return (state_oper * super()._default_start_state()).unit()

class SinglePhotonEmission(Protocol):
	def __init__(self, name:str, parameters: dict, dim:int=3, *args, **kwargs):

		super().__init__(name, parameters, dim=dim, *args, **kwargs)

		self.photon_names_incoherent: [str] = [x+str('_incoherent') for x in self.photon_names]
		self.photon_modes_incoherent: [nq.NQobj] = [x.rename(x.name(), x.name()+str('_incoherent')) for x in self.photon_modes]

	# def _perpare_photons_incoherent(self) -> None:
	# 	"""
	# 	Prepares vacuum photon modes for each photon name present
	# 	"""

	# 	for name in self.photon_names_incoherent:
	# 		self.photon_modes_incoherent.append(
	# 					nq.name(vacuum_dim(self.dim), name, kind='state')
	# 				)

	def _prepare_blank_dm(self) -> None:
		"""
		prepares a "blank" (identity) density matrix with the right mode names and dimensions
		"""
		#photon_dm = [nq.ket2dm(x) for x in self.photon_modes]
		photon_dm = [nq.name(qt.qeye(self.dim), x, 'state') for x in self.photon_names]
		self.dm = nq.tensor( self.alice, self.bob, *photon_dm)

	def _default_start_state(self) -> nq.NQobj:
		""" 
		Sets the default start state. Must be defined by subclasses according to the specific protocol 
		"""

		return nq.tensor( nq.name(x, 'Alice', 'state'), nq.name(x, 'Bob', 'state'), *self.photon_modes).unit()

	def _default_start_state(self):
		"""
		Start state has Alice and Bob in x, and photon in (E+L)
		"""
		if self.parameters['alpha'] is None:
			creators = nq.tensor( 
						(nq.name(create_dim(self.dim), self.photon_names[0], 'oper') +\
						 nq.name(create_dim(self.dim), self.photon_names[1], 'oper')),
						 (nq.name(create_dim(self.dim), self.photon_names[2], 'oper') +\
						 nq.name(create_dim(self.dim), self.photon_names[3], 'oper'))
						 )
		else:
			creators = nq.tensor( 
						(nq.name(displace_dim(self.dim, self.parameters['alpha']), self.photon_names[0], 'oper') +\
						 nq.name(displace_dim(self.dim, self.parameters['alpha']), self.photon_names[1], 'oper')),
						 (nq.name(displace_dim(self.dim, self.parameters['alpha']), self.photon_names[2], 'oper') +\
						 nq.name(displace_dim(self.dim, self.parameters['alpha']), self.photon_names[3], 'oper'))
						 )
		
		state_oper = nq.tensor( nq.name(qt.qeye(2), 'Alice', 'oper'), nq.name(qt.qeye(2), 'Bob', 'oper'), creators)
		return (state_oper * super()._default_start_state()).unit()
