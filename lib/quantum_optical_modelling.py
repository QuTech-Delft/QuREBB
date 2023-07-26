import numpy as np

def cavity_Rempe(N, kappa_r, kappa_t, kappa_m, gamma, g):
    kappa = kappa_r + kappa_t + kappa_m # below eq. 10
    D = N*g**2 + kappa*gamma # Denominator in equations
    r = (N*g**2 + (kappa - 2*kappa_r) * gamma)/D # eq. 7
    t = (2*np.sqrt(kappa_r*kappa_t)*gamma)/D # eq. 8
    m = (2*np.sqrt(kappa_r*kappa_m)*gamma)/D # eq. 9
    a = (2*np.sqrt(kappa_r*gamma)*np.sqrt(N)*g)/D # eq. 10
    l = np.sqrt(m**2 + a**2)
    return t, r, l


def cavity_qom(delta_al, delta_ac, delta_cl, kappa_r, kappa_t, kappa_loss, gamma, C, gamma_dephasing=0):
    kappa_tot=kappa_r+kappa_t+kappa_loss
    gamma_tot=gamma+gamma_dephasing

    C_eff= C #/(1+2*delta_ac**2/kappa_tot**2)

    r= 1- kappa_r/kappa_tot / (-1j *delta_cl/kappa_tot + 0.5 + C_eff/(-1j*delta_al/gamma_tot + 0.5))
    t= kappa_t/kappa_tot / (-1j*delta_cl/kappa_tot + 0.5 + C_eff/(-1j*delta_al/gamma_tot + 0.5))
    l= np.sqrt(1-np.abs(t)**2 - np.abs(r)**2)

    return t,r,l

def cavity_qom_atom_centered(omega,delta, kappa_r, kappa_t, kappa_loss, gamma, C, gamma_dephasing=0):

    delta_ac=delta
    delta_cl=omega+delta
    delta_al=omega

    t,r,l= cavity_qom(delta_al, delta_ac, delta_cl, kappa_r, kappa_t, kappa_loss, gamma, C, gamma_dephasing)
    return t,r,l

def cavity_qom_cavity_centered(omega,delta, kappa_r, kappa_t, kappa_loss, gamma, C, gamma_dephasing=0):

    delta_ac=delta
    delta_cl=omega
    delta_al=delta+omega

    t,r,l= cavity_qom(delta_al, delta_ac, delta_cl, kappa_r, kappa_t, kappa_loss, gamma, C, gamma_dephasing)
    return t,r,l

def cavity_qom_atom_centered_controlled(omega,delta, kappa_r, kappa_t, kappa_loss, gamma, C, N=0, gamma_dephasing=0):

    delta_ac=delta
    delta_cl=omega-delta
    delta_al=omega

    t,r,l= cavity_qom(delta_al, delta_ac, delta_cl, kappa_r, kappa_t, kappa_loss, gamma, N*C, gamma_dephasing)
    return t,r,l

def cavity_enhanced_incoherent_emission(kappa_r, kappa_t, gamma, gamma_dephasing, DW, C):
    gamma_r = gamma*DW
    p_coherent = kappa_r/(kappa_r+kappa_t) * C/(C+1) * (C/DW)*gamma_r/(gamma+gamma_dephasing+(C/DW)*gamma_r)
    p_incoherent = kappa_r/(kappa_r+kappa_t) * C/(C+1) * gamma_dephasing/(gamma+gamma_dephasing+(C/DW)*gamma_r)
    p_2ph = 0
    p_loss = 1 - p_coherent - p_incoherent - p_2ph
    return p_coherent, p_incoherent, p_2ph, p_loss

