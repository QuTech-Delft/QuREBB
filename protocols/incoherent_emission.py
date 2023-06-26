## Logical building blocks and tools for emission based scheme protocol (1 and 2 photons)

from lib.states import *
import lib.NQobj as nq
import lib.PBB as pbb
import lib.quantum_optical_modelling as qom
import lib.LBB as lbb

import numpy as np
import qutip as qt
import scipy.constants as cst
import matplotlib.pyplot as plt


## Prepare protocol initial state

def PrepareSingleClickIdeal(protocol_params, dim=3):

	alpha = protocol_params['alpha']

	spin_init =  (np.sqrt(alpha)*down + np.sqrt(1-alpha)*up).unit()

	alice_init = nq.name(spin_init, 'Alice')
	bob_init = nq.name(spin_init, 'Bob')

	ph_alice = nq.name(vacuum_dim(dim), 'ph_alice')
	ph_bob = nq.name(vacuum_dim(dim), 'ph_bob')

	dm_in = nq.ket2dm(nq.tensor(alice_init, bob_init, ph_alice, ph_bob))

	return dm_in

def PrepareSingleClickNonIdeal(protocol_params, dim=3):

	alpha = protocol_params['alpha']

	spin_init =  (np.sqrt(alpha)*down + np.sqrt(1-alpha)*up).unit()

	alice_init = nq.name(spin_init, 'Alice')
	bob_init = nq.name(spin_init, 'Bob')

	ph_alice = nq.name(vacuum_dim(dim), 'ph_alice')
	ph_bob = nq.name(vacuum_dim(dim), 'ph_bob')
	ph_alice_incoh = nq.name(vacuum_dim(dim), 'ph_alice_incoh')
	ph_bob_incoh = nq.name(vacuum_dim(dim), 'ph_bob_incoh')

	dm_in = nq.ket2dm(nq.tensor(alice_init, bob_init, ph_alice, ph_bob, ph_alice_incoh, ph_bob_incoh))

	return dm_in

def PrepareDoubleClick(dim=3):

	alice_init = nq.name(x, 'Alice')
	bob_init = nq.name(x, 'Bob')

	E_alice = nq.name(vacuum_dim(dim), 'E_alice')
	L_alice = nq.name(vacuum_dim(dim), 'L_alice')

	E_bob = nq.name(vacuum_dim(dim), 'E_bob')
	L_bob = nq.name(vacuum_dim(dim), 'L_bob')

	dm_in = nq.ket2dm(nq.tensor(alice_init, bob_init, E_alice, L_alice, E_bob, L_bob))

	return dm_in




#Protocols

def DoubleClickProtocol(protocol_params):

	dim=kwargs.pop('dim', protocol_params.get('dim', 3))

	E_alice = nq.name(vacuum_dim(dim), 'E_alice')
	L_alice = nq.name(vacuum_dim(dim), 'L_alice')

	E_bob = nq.name(vacuum_dim(dim), 'E_bob')
	L_bob = nq.name(vacuum_dim(dim), 'L_bob')

	dm_init = PrepareDoubleClick()

	dm_1 = lbb.EmissionTimeBinSPI(dm_init, 'Alice', ['E_alice', 'L_alice'], dim=dim)

	dm_2 = lbb.EmissionTimeBinSPI(dm_1, 'Bob', ['E_bob', 'L_bob'], dim=dim)

	dm_phase = lbb.Phase(dm_2, protocol_params, 'ph_bob', dim=dim)

	dm_3 = lbb.HOM(dm_phase, ['E_alice', 'E_bob'], dim=dim)

	dm_4 = lbb.HOM(dm_3, ['L_alice', 'L_bob'], dim=dim)

	herald_state = nq.tensor( nq.tensor(nq.name(qt.qeye(dim), 'E_alice', 'oper'), nq.name(qt.qeye(dim), 'L_alice', 'oper') ) -\
					nq.tensor(E_alice.proj(), L_alice.proj()),\
					nq.tensor(nq.name(qt.qeye(dim), 'E_bob', 'oper'), nq.name(qt.qeye(dim), 'L_bob', 'oper') ) -\
					nq.tensor(E_bob.proj(), L_bob.proj()) )

	dm_5 = lbb.MeasureProj(dm_4, herald_state2)

	target_state= (nq.tensor(nq.name(up, 'Alice'), nq.name(down, 'Bob')) -\
			nq.tensor(nq.name(down, 'Alice'), nq.name(up, 'Bob'))).unit()

	return lbb.Metrics(dm_5, target_state)


def SingleClickProtocolIdeal(protocol_params, **kwargs):

	dim=kwargs.pop('dim', protocol_params.get('dim', 3))

	if protocol_params['alpha'] is None:
		print('Bright state population alpha was not provided')
		return

	ph_alice = nq.name(vacuum_dim(dim), 'ph_alice')
	ph_bob = nq.name(vacuum_dim(dim), 'ph_bob')

	dm_in = PrepareSingleClickIdeal(protocol_params, dim=dim)

	dm_1 = lbb.EmissionOnePhotonSPI(dm_in, 'Alice', 'ph_alice', dim=dim)

	dm_2 = lbb.EmissionOnePhotonSPI(dm_1, 'Bob', 'ph_bob', dim=dim)

	dm_phase1 = lbb.ModeLoss(dm_2, 'ph_alice', protocol_params, par_name='link_loss', dim=dim)
	dm_phase = lbb.ModeLoss(dm_phase1, 'ph_bob', protocol_params, par_name='link_loss', dim=dim)

	dm_3 = lbb.HOM(dm_phase, ['ph_alice', 'ph_bob'], dim=dim)

	herald_state = nq.name(photon_dim(dim).proj(), 'ph_alice', 'oper')

	dm_4 = lbb.DarkCounts(dm_3, protocol_params, 'ph_alice', dim=dim)
	dm_5 = lbb.DarkCounts(dm_4, protocol_params, 'ph_bob', dim=dim)

	dm_6 = lbb.MeasureProj(dm_5, herald_state)

	target_state= (nq.tensor(nq.name(up, 'Alice'), nq.name(down, 'Bob')) +\
	           nq.tensor(nq.name(down, 'Alice'), nq.name(up, 'Bob'))).unit()

	return lbb.Metrics(dm_6, target_state)


def SingleClickProtocol(protocol_params, **kwargs):

	dim=kwargs.pop('dim', protocol_params.get('dim', 3))

	if protocol_params['alpha'] is None:
		print('Bright state population alpha was not provided')
		return

	ph_alice = nq.name(vacuum_dim(dim), 'ph_alice')
	ph_bob = nq.name(vacuum_dim(dim), 'ph_bob')
	ph_alice_incoh = nq.name(vacuum_dim(dim), 'ph_alice_incoh')
	ph_bob_incoh = nq.name(vacuum_dim(dim), 'ph_bob_incoh')

	dm_in = PrepareSingleClickNonIdeal(protocol_params, dim=dim)

	dm_1 = lbb.EmissionOnePhotonSPINonIdeal(dm_in, protocol_params, 'Alice', 'ph_alice', 'ph_alice_incoh', dim=dim)

	dm_1_insertion_loss = lbb.ModeLoss(dm_1, 'ph_alice', protocol_params, par_name = 'insertion_loss', dim=dim)
	dm_1_insertion_loss_incoh = lbb.ModeLoss(dm_1_insertion_loss, 'ph_alice_incoh', protocol_params, par_name = 'insertion_loss', dim=dim)

	dm_2 = lbb.EmissionOnePhotonSPINonIdeal(dm_1_insertion_loss_incoh, protocol_params, 'Bob', 'ph_bob', 'ph_bob_incoh', dim=dim)

	dm_2_insertion_loss = lbb.ModeLoss(dm_2, 'ph_bob', protocol_params, par_name = 'insertion_loss', dim=dim)
	dm_2_insertion_loss_incoh = lbb.ModeLoss(dm_2_insertion_loss, 'ph_bob_incoh', protocol_params, par_name = 'insertion_loss', dim=dim)

	dm_loss_a = lbb.ModeLoss(dm_2_insertion_loss_incoh, 'ph_alice', protocol_params, par_name='link_loss', dim=dim)
	dm_loss_a_inc = lbb.ModeLoss(dm_loss_a, 'ph_alice_incoh', protocol_params, par_name='link_loss', dim=dim)
	dm_loss_b = lbb.ModeLoss(dm_loss_a_inc, 'ph_bob', protocol_params, par_name='link_loss', dim=dim)
	dm_loss_b_inc = lbb.ModeLoss(dm_loss_b, 'ph_bob_incoh', protocol_params, par_name='link_loss', dim=dim)

	dm_3 = lbb.HOM(dm_loss_b_inc, ['ph_alice', 'ph_bob'], dim=dim)

	dm_incoh_mode =  nq.tensor(nq.ket2dm(nq.name(vacuum_dim(dim), 'temporary_incoherent_mode_1')), dm_3)

	dm_incoh_hom = lbb.HOM(dm_incoh_mode, ['ph_alice_incoh', 'temporary_incoherent_mode_1'], dim=dim)

	dm_hom =  dm_incoh_hom.ptrace(['temporary_incoherent_mode_1'], keep=False)

	herald_state = nq.tensor(nq.name(qt.qeye(dim), 'ph_alice', 'oper'),nq.name(qt.qeye(dim), 'ph_alice_incoh', 'oper')) -\
					 nq.tensor(ph_alice.proj(), ph_alice_incoh.proj())


	dm_4 = lbb.DarkCounts(dm_hom, photon_names='ph_alice', parameters=protocol_params, dim=dim)
	dm_5 = lbb.DarkCounts(dm_4, photon_names='ph_bob', parameters=protocol_params,  dim=dim)

	dm_6 = lbb.MeasureProj(dm_5, herald_state)

	target_state= (nq.tensor(nq.name(up, 'Alice'), nq.name(down, 'Bob')) +\
	           nq.tensor(nq.name(down, 'Alice'), nq.name(up, 'Bob'))).unit()

	return lbb.Metrics(dm_6, target_state)

