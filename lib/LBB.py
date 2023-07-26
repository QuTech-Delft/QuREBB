from lib.states import *
import lib.NQobj as nq
import lib.PBB as pbb
import lib.quantum_optical_modelling as qom

import numpy as np
import qutip as qt
import scipy.constants as cst
import matplotlib.pyplot as plt


###########
## SPIs  ##
###########


def EmissionOnePhotonSPI(dm_in, spin_name, photon_name, dim=2):
    
    #define building block
    
    emitter = pbb.emitter_pi_pulse(dim=dim)
    emitter.rename('spin', spin_name)
    emitter.rename('photon', photon_name)
    
    dm_out = emitter*dm_in*emitter.dag()
    
    return dm_out

def EmissionOnePhotonSPINonIdeal(dm_in, protocol_params, spin_name, photon_name, incoherent_name, dim=2):
    #define building block
    #cavity params must be [kappa_in, kappa_loss, gamma, g or C]
    kappa_r=protocol_params['kappa_in']
    kappa_t=protocol_params['kappa_loss']
    gamma=protocol_params['gamma']

    
    DW = protocol_params['DW']
    QE = protocol_params['QE']
    if 'gamma_dephasing' in protocol_params:
        gamma_dephasing=protocol_params['gamma_dephasing']
    else:
        gamma_dephasing=0

    if 'C' in protocol_params:
        C=protocol_params['C']
    elif 'g' in protocol_params:
        g=protocol_params['g']
        C= 4*g**2/(kappa_t+kappa_r)/(gamma+gamma_dephasing)
    else:
        print('no C or g found in parameters. Using defaulr C=0.01')
        C=0.01

    p_coh, p_incoh, p_2ph, p_loss = qom.cavity_enhanced_incoherent_emission(kappa_r, kappa_t, gamma, gamma_dephasing, DW*QE, C)
     
    c_coh = pbb.emitter_pi_pulse(dim=dim)
    c_coh.rename('spin', spin_name)
    c_coh.rename('photon', photon_name)

    c_loss = pbb.emitter_error(dim=dim)
    c_loss.rename('spin', spin_name)
    c_loss.rename('photon', 'loss')

    c_incoh = pbb.emitter_error(dim=dim)
    c_incoh.rename('spin', spin_name)
    c_incoh.rename('photon', incoherent_name)

    c_2ph = pbb.emitter_twophoton_error(dim=dim)
    c_2ph.rename('spin', spin_name)
    c_2ph.rename('photon', photon_name)

    C = (np.sqrt(p_coh)*c_coh + np.sqrt(p_loss)*c_loss)
    dm_out = C*dm_in*C.dag() + p_incoh*c_incoh*dm_in*c_incoh.dag() + p_2ph * c_2ph * dm_in * c_2ph.dag()
    
    return nq.trace_out_loss_modes(dm_out)


def EmissionTimeBinSPI(dm_in, spin_name, photon_names, dim=2, **kwargs):
    
    #define building block    
    emitter_E = pbb.emitter_pi_pulse(dim=dim)
    emitter_E.rename('spin', spin_name)
    emitter_E.rename('photon', photon_names[0])
    
    emitter_L = emitter_pi_pulse(dim=dim)
    emitter_L.rename('spin', spin_name)
    emitter_L.rename('photon', photon_names[1])
    
    RX_pi  = nq.name([[0, -1j], [-1j, 0]], spin_name)
    
    #Apply sequence:
    dm_E = emitter_E*dm_in*emitter_E.dag()
    dm_X = RX_pi*dm_E*RX_pi.dag()
    dm_L = emitter_L*dm_X*emitter_L.dag()
    
    return dm_L


def CavityTimeBinSPI(dm_in, spin_name=None, photon_names=None, parameters=None, dim=2, **kwargs):

    #cavity params must be [f, kappa_in, kappa_loss, gamma, delta, splitting, g]
    atom_centered = kwargs.pop('atom_centered', True)
    f_oper = parameters['f_operation']
    kappa_r=parameters['kappa_in']
    kappa_t=parameters['kappa_loss']
    kappa_loss=0
    gamma=parameters['gamma']
    delta=parameters['delta']
    splitting=parameters['splitting']
    g=parameters['g']
    f_oper=parameters['f_operation']

    if 'gamma_dephasing' in parameters:
        gamma_dephasing=parameters['gamma_dephasing']
    else:
        gamma_dephasing=0

    #print('tf')
    if 'C' in parameters and parameters['C'] is not None:
        C=parameters['C']
        #print(f"C is {C}")
    else:
        C = 4*g**2/(kappa_t+kappa_r+kappa_loss)/(gamma+gamma_dephasing)
        #print(f"C not found, using {C}")

    if atom_centered:
        t_u, r_u, l_u = qom.cavity_qom_atom_centered(f_oper, -delta, kappa_r, kappa_t, kappa_loss, gamma, C, gamma_dephasing=gamma_dephasing)
        t_d, r_d, l_d = qom.cavity_qom_atom_centered(f_oper+splitting/2, -delta-splitting/2, kappa_r, kappa_t, kappa_loss, gamma, C, gamma_dephasing=gamma_dephasing)


    else:
        t_u, r_u, l_u = qom.cavity_qom_cavity_centered(f_oper, delta, kappa_r, kappa_t, kappa_loss, gamma, C, gamma_dephasing=gamma_dephasing)
        t_d, r_d, l_d = qom.cavity_qom_cavity_centered(f_oper, delta-splitting, kappa_r, kappa_t, kappa_loss, gamma, C, gamma_dephasing=gamma_dephasing)

    loss_u = np.sqrt(np.abs(t_u)**2 + np.abs(l_u)**2)
    loss_d = np.sqrt(np.abs(t_d)**2 + np.abs(l_d)**2)

    cav = pbb.cavity_single_sided_unitary(r_u, loss_u, r_d, loss_d , dim=dim)
    cav.rename('spin', spin_name)

    #Apply early
    cav.rename('R'   , photon_names[0] )
    cav.rename('loss', 'loss1')

    dm_E_full = cav*dm_in*cav.dag()
    dm_E = nq.trace_out_loss_modes(dm_E_full)

    #Flip spin and scatter late
    RX_pi  = nq.NQobj([[0, -1j], [-1j, 0]], names=spin_name)
    cav.rename(photon_names[0]   , photon_names[1]    )
    cav.rename('loss1', 'loss2')
    
    dm_L_full = cav * (RX_pi * dm_E * RX_pi.dag()) * cav.dag()
    #dm_L_full = RX_pi * (cav * (RX_pi * dm_E * RX_pi.dag()) * cav.dag() ) * RX_pi.dag()
    dm_L = nq.trace_out_loss_modes(dm_L_full)
    
    return dm_L

def CavityPhaseSPI(dm_in, spin_name=None, photon_names=None, parameters=None, dim=2, **kwargs):

    #cavity params must be [f, kappa_in, kappa_loss, gamma, delta, splitting, g]
    f_oper = parameters['f_operation']
    kappa_r=parameters['kappa_in']
    kappa_t=parameters['kappa_loss']
    kappa_loss=0
    gamma=parameters['gamma']
    delta=parameters['delta']
    splitting=parameters['splitting']
    g=parameters['g']
    N = parameters['N']
    gamma_dephasing=parameters.get('gamma_dephasing', 0)

    if 'C' in parameters and parameters['C'] is not None:
        C=parameters['C']
        print(f"C is {C}")
    else:
        C = 4*g**2/(kappa_t+kappa_r+kappa_loss)/(gamma+gamma_dephasing)
        print(f"C not found, using {C}")

    f_oper=parameters['f_operation']

    if 'gamma_dephasing' in parameters:
        gamma_dephasing=parameters['gamma_dephasing']
    else:
        gamma_dephasing=0

    t_on, r_on, l_on = qom.cavity_qom_atom_centered(f_oper, delta, kappa_r, kappa_t, kappa_loss, gamma, C, gamma_dephasing=gamma_dephasing)
    t_off, r_off, l_off = qom.cavity_qom_atom_centered(f_oper, delta, kappa_r, kappa_t, kappa_loss, gamma, 0*C, gamma_dephasing=gamma_dephasing)

    loss_on = np.sqrt(np.abs(t_on)**2 + np.abs(l_on)**2) # np.sqrt(1-np.abs(r_u)**2) #np.sqrt(t_u**2 + l_u**2)
    loss_off = np.sqrt(np.abs(t_off)**2 + np.abs(l_off)**2) #np.sqrt(1-np.abs(r_d)**2) #np.sqrt(t_d**2 + l_d**2)

    #For R-polarised light the cavity response is spin dependent (ON for up and OFF for down)
    cav_R = pbb.cavity_single_sided_unitary(r_on, loss_on, r_off, loss_off, dim=dim)   
    cav_R.rename('spin', spin_name)
    cav_R.rename('R', photon_names[0])
    cav_R.rename('loss', 'loss1')

    #For L-polarised light the cavity is always off
    cav_L = pbb.cavity_single_sided_unitary(r_off, loss_off, r_off, loss_off, dim=dim)   
    cav_L.rename('spin', spin_name)
    cav_L.rename('R', photon_names[1])
    cav_L.rename('loss', 'loss2')

    #Apply cavity
    dm_out_full = cav_L* (cav_R*dm_in*cav_R.dag()) *cav_L.dag()
    dm_out = dm_out_full.ptrace(['loss1', 'loss2'], keep=False)
    return dm_out

#######################
## Photon operations ##
#######################

def HOM(dm_in, photon_names, dim=3, **kwargs):
    
    hom_bs = pbb.unitary_beamsplitter(theta=np.pi/4, dim=dim)
    hom_bs.rename('A', photon_names[0])
    hom_bs.rename('B', photon_names[1])
    
    return hom_bs*dm_in*hom_bs.dag()


def BasisRotation(dm_in, photon_names=None, sign=+1, dim=2, **kwargs):
    wp=pbb.unitary_beamsplitter(sign*np.pi/4, dim=dim)
    wp.rename('A', photon_names[0])
    wp.rename('B', photon_names[1])

    return wp*dm_in*wp.dag()

def ModeLoss(dm_in, photon_names=None, parameters = None, dim=2, **kwargs):

    par_name = kwargs.pop('par_name', 'loss')
    loss_factor = parameters.get(par_name, 0.)

    #add loss along the way
    link_loss = pbb.loss(loss_factor, dim=dim)
    link_loss.rename('A', photon_names)
    dm_loss = link_loss * dm_in * link_loss.dag()

    dm_out = dm_loss.ptrace(['loss'], keep=False)
    return dm_out


###############
##  Sources  ##
###############

def SinglePhotonSource(dm_in, photon_name):
	pass

#################
##  Spin gates ##
#################

def SpinPiX(dm_in, spin_name, **kwargs):
	RX_pi  =  nq.name([[0, -1j], [-1j, 0]], names=spin_name)
	return RX_pi*dm_in*RX_pi.dag()

def SpinPiY(dm_in, spin_name, **kwargs):
	RY_pi  =  nq.name([[0,  -1], [  1, 0]], names=spin_name)
	return RY_pi*dm_in*RY_pi.dag()

########################
##  Detect & Measure  ##
########################

def PhotonDetection(dm_in, spin_name, photon_names, photon_basis):
    detect = photon_basis[0].proj() + photon_basis[1].proj()
    dm_measured = detect * dm_in.ptrace([spin_name, photon_names[0], photon_names[1]])*detect.dag()
    return dm_measured

def PhotonDetectionProj(dm_in, state_projectors, spin_name, photon_names):
    detect = state_projectors[0] + state_projectors[1]
    dm_measured = detect * dm_in.ptrace([spin_name, photon_names[0], photon_names[1]])*detect.dag()
    return dm_measured

def Measure(dm_in, herald_state):
    herald = herald_state.proj()
    dm_final = (herald * dm_in * herald.dag())
    return nq.trace_out_everything_but_spins(dm_final)

def MeasureProj(dm_in, herald_proj):
    dm_final = herald_proj * dm_in * herald_proj.dag()
    return nq.trace_out_everything_but_spins(dm_final)
    
def Metrics(dm_in, target_state):
    rate = dm_in.tr()
    target_dm = nq.ket2dm(target_state)
    fid = nq.fidelity(dm_in.unit(), target_dm)**2
    return fid, rate



def ConditionalGateMeasurement(dm_in, states, target_spin_name):
	dm_out= states[0].proj()*dm_in*states[0].proj().dag() + SpinPiX( SpinPiY(states[1].proj()*dm_in*states[1].proj().dag(), target_spin_name), target_spin_name)
	return nq.trace_out_everything_but_spins(dm_out)

def ConditionalGateMeasurementProj(dm_in, state_projectors, target_spin_name):
	dm_out= state_projectors[0]*dm_in*state_projectors[0].dag() + SpinPiX( SpinPiY(state_projectors[1]*dm_in*state_projectors[1].dag(), target_spin_name), target_spin_name)
	return nq.trace_out_everything_but_spins(dm_out)


############################
##  Noise & Imperfections ##
############################


def DarkCounts(dm_in, photon_names=None, parameters=None, dim=3, **kwargs):
    dc_rate = parameters['dc_rate']

    a = nq.name(qt.destroy(dim), photon_names)
    I = nq.name(qt.qeye(dim), photon_names, 'oper')

    return (dc_rate)*(a.dag()*dm_in*a) + (1-dc_rate)*dm_in


def Phase(dm_in, protocol_params, photon_mode, dim=3):

    try:
        pha = protocol_params['phase']
    except:
        pha = 0.

    phase_shifter = pbb.phase(pha, dim=dim)
    phase_shifter.rename('photon', photon_mode)
    return phase_shifter*dm_in*phase_shifter.dag()











