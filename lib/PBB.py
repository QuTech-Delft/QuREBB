import numpy as np
import qutip as qt
from . import NQobj as nq
from .states import *
import math
from functools import lru_cache

def cavity(r_u, t_u, l_u, r_d, t_d, l_d):
    tot_d = np.abs(t_d)**2 + np.abs(r_d)**2 + np.abs(l_d)**2 
    tot_u = np.abs(t_u)**2 + np.abs(r_u)**2 + np.abs(l_u)**2
    if not math.isclose(tot_d, 1, abs_tol=1e-6):
        raise ValueError('The squares of r_d, t_d and l_d should add up to 1.') 
    if not math.isclose(tot_u, 1, abs_tol=1e-6):
        raise ValueError('The squares of r_u, t_u and l_u should add up to 1.')

    R     = nq.name(photon, 'R'   , kind='state')
    R_vac = nq.name(vacuum, 'R'   , kind='state')
    T     = nq.name(photon, 'T'   , kind='state')
    T_vac = nq.name(vacuum, 'T'   , kind='state')
    L     = nq.name(photon, 'loss', kind='state')
    L_vac = nq.name(vacuum, 'L'   , kind='state')
    u     = nq.name(up    , 'spin', kind='state')
    d     = nq.name(down  , 'spin', kind='state')

    cav =\
    nq.tensor(u, r_u*R + t_u*T + l_u*L) * nq.tensor(u, R).dag() +\
    nq.tensor(d, r_d*R + t_d*T + l_d*L) * nq.tensor(d, R).dag() +\
    nq.tensor(R_vac, T_vac, L_vac) * R_vac.dag()
    return cav

def cavity_unitary(r_u, t_u, l_u, r_d, t_d, l_d, dim=2):
    tot_d = np.abs(t_d)**2 + np.abs(r_d)**2 + np.abs(l_d)**2 
    tot_u = np.abs(t_u)**2 + np.abs(r_u)**2 + np.abs(l_u)**2
    if not math.isclose(tot_d, 1, abs_tol=1e-6):
        raise ValueError('The squares of r_d, t_d and l_d should add up to 1.') 
    if not math.isclose(tot_u, 1, abs_tol=1e-6):
        raise ValueError('The squares of r_u, t_u and l_u should add up to 1.')

    R     = nq.name(qt.destroy(dim), 'R'   )
    T     = nq.name(qt.destroy(dim), 'T'   )
    L     = nq.name(qt.destroy(dim), 'loss')
    u     = nq.name(up    , 'spin')
    d     = nq.name(down  , 'spin')

    theta_loss_u = np.arctan( np.sqrt((1-l_u)/(l_u)) )
    theta_splitting_u = np.arctan( r_u/t_u )
    cav_u = ( 1j* (np.angle(r_u)*R.dag()*R) ).expm() *\
            ( 1j* (np.angle(t_u)*T.dag()*T) ).expm() *\
            ( theta_splitting_u * (R.dag()*T - R*T.dag()) ).expm() *\
            ( theta_loss_u * (R.dag()*L - R*L.dag()) ).expm() 

    theta_loss_d = np.arctan( np.sqrt((1-l_d)/l_d) )
    theta_splitting_d = np.arctan( r_d/t_d )
    cav_d = ( 1j* (np.angle(r_d)*R.dag()*R) ).expm() *\
            ( 1j* (np.angle(t_d)*T.dag()*T) ).expm() *\
            ( theta_splitting_d * (R.dag()*T - R*T.dag()) ).expm() *\
            ( theta_loss_d * (R.dag()*L - R*L.dag()) ).expm() 
    cav_tot=nq.tensor(u.proj(), cav_u) + nq.tensor(d.proj(), cav_d)
    return cav_tot

def cavity_single_sided(r_u, l_u, r_d, l_d):
    tot_d = np.abs(r_d)**2 + np.abs(l_d)**2 
    tot_u = np.abs(r_u)**2 + np.abs(l_u)**2
    if not math.isclose(tot_d, 1, abs_tol=1e-6):
        raise ValueError('The squares of r_d and l_d should add up to 1.') 
    if not math.isclose(tot_u, 1, abs_tol=1e-6):
        raise ValueError('The squares of r_u and l_u should add up to 1.')
    
    R     = nq.name(photon, 'R'   )
    R_vac = nq.name(vacuum, 'R'   )
    L     = nq.name(photon, 'loss')
    L_vac = nq.name(vacuum, 'loss'  )
    u     = nq.name(up    , 'spin')
    d     = nq.name(down  , 'spin')

    cav =\
    nq.tensor(u, r_u*R + l_u*L) * nq.tensor(u, R).dag() +\
    nq.tensor(d, r_d*R + l_d*L) * nq.tensor(d, R).dag() +\
    nq.tensor(R_vac, L_vac) * R_vac.dag()
    return cav


def cavity_single_sided_unitary(r_u, l_u, r_d, l_d, dim=2):
    tot_d = np.abs(r_d)**2 + np.abs(l_d)**2 
    tot_u = np.abs(r_u)**2 + np.abs(l_u)**2
    if not math.isclose(tot_d, 1, abs_tol=1e-6):
        raise ValueError('The squares of r_d and l_d should add up to 1.') 
    if not math.isclose(tot_u, 1, abs_tol=1e-6):
        raise ValueError('The squares of r_u and l_u should add up to 1.')
    
    R     = nq.name(qt.destroy(dim), 'R'   )
    L     = nq.name(qt.destroy(dim), 'loss')
    u     = nq.name(up    , 'spin')
    d     = nq.name(down  , 'spin')

    theta_loss_u = np.arctan( np.abs(l_u)/np.abs(r_u) )
    cav_u = ( 1j* (np.angle(r_u)*R.dag()*R) ).expm() *\
            ( 1j* (np.angle(l_u)*L.dag()*L) ).expm() *\
            ( theta_loss_u * (L.dag()*R - L*R.dag()) ).expm()

    theta_loss_d = np.arctan( np.abs(l_d)/np.abs(r_d) )
    cav_d = ( 1j* (np.angle(r_d)*R.dag()*R) ).expm() *\
            ( 1j* (np.angle(l_d)*L.dag()*L) ).expm() *\
            ( theta_loss_d * (L.dag()*R - L*R.dag()) ).expm()

    return nq.tensor(u.proj(), cav_u) + nq.tensor(d.proj(), cav_d)


def beamsplitter_single_input(transmission=.5):
    A     = nq.name(photon, 'A')
    A_vac = nq.name(photon, 'A')
    B     = nq.name(photon, 'B')
    B_vac = nq.name(photon, 'B')

    BS = \
    nq.tensor(np.sqrt(transmission)*A + np.sqrt(1-transmission)*B) * A.dag()# +\
    nq.tensor(A_vac, B_vac) * A_vac.dag()
    return BS

def unitary_beamsplitter(theta=0, dim=2):
    A = nq.name(qt.destroy(dim), 'A')
    B = nq.name(qt.destroy(dim), 'B')
    #return (1j*theta*(A.dag()*B + A*B.dag())).expm()
    return (theta*(A.dag()*B - A*B.dag())).expm()


def loss(loss=0.5, dim=2):
    A = nq.name(qt.destroy(dim), 'A')
    L = nq.name(qt.destroy(dim), 'loss')
    theta= np.arctan( np.sqrt( (loss)/(1-loss) ) )
    return ( theta * (A.dag()*L - A*L.dag()) ).expm()



def waveplate(theta=0, dim=2):
    r=nq.name(qt.destroy(dim), 'H')
    l=nq.name(qt.destroy(dim), 'V')

    return (-1j*theta*(r.dag()*l + r*l.dag())).expm()


def emitter_pi_pulse(dim=2):
    
    photon = nq.name(qt.destroy(dim), 'photon')
    I = nq.name(qt.qeye(dim), 'photon', 'oper')
    u = nq.name(up, 'spin')
    d = nq.name(down, 'spin')
    
    return nq.tensor(u.proj(), I) + nq.tensor(d.proj(), photon.dag())

def emitter_pi_pulse_weighted(etas,dim=2):

    photons = []
    Is = []
    
    for i,e in enumerate(etas):
        photons.append(nq.name(qt.destroy(dim), 'photon_{}'.format(i)))
        Is.append(nq.name(qt.qeye(dim), 'photon_{}'.format(i), 'oper'))
    u = nq.name(up, 'spin')
    d = nq.name(down, 'spin')
    dm = nq.tensor(u.proj(), *Is)
    for i,e in enumerate(etas):
        dm += e*nq.tensor(d.proj(), photons[i].dag())
    return  dm

def emitter_error(dim=2):
    photon = nq.name(qt.destroy(dim), 'photon')
    I = nq.name(qt.qeye(dim), 'photon', 'oper')
    u = nq.name(up, 'spin')
    d = nq.name(down, 'spin')
    
    return nq.tensor(d.proj(), photon.dag())


def emitter_loss(dim=2):
    
    photon = nq.name(qt.destroy(dim), 'photon')
    I = nq.name(qt.qeye(dim), 'photon', 'oper')
    u = nq.name(up, 'spin')
    d = nq.name(down, 'spin')
    
    return nq.tensor(u.proj(), I) + nq.tensor(d.proj(), I)

def emitter_twophoton(dim=3):
    photon = nq.name(qt.destroy(dim), 'photon')
    I = nq.name(qt.qeye(dim), 'photon', 'oper')
    u = nq.name(up, 'spin')
    d = nq.name(down, 'spin')
    
    return (nq.tensor(u.proj(), I) + nq.tensor(d.proj(), photon.dag()**2/np.sqrt(2)))

def emitter_twophoton_error(dim=3):
    photon = nq.name(qt.destroy(dim), 'photon')
    I = nq.name(qt.qeye(dim), 'photon', 'oper')
    u = nq.name(up, 'spin')
    d = nq.name(down, 'spin')
    
    return nq.tensor(d.proj(), photon.dag()**2/np.sqrt(2))

def phase(theta=0,dim=2):
    a = nq.name(qt.destroy(dim), 'photon')
    return (1j*theta*a.dag()*a).expm()