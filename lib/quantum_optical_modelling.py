import numpy as np


def cavity_qom(delta_al, delta_ac, delta_cl, kappa_r, kappa_t, kappa_loss, gamma, C, gamma_dephasing=0):
    """
    Generic quantum optical model of cavity. Ref?

    Parameters:
        delta_al: detuning between atom (emitter) and probe laser
        delta_ac: detuning between atom (emitter) and cavity
        delta_cl: detuning between cavity and probe laser
        kappa_r:  cavity decay rate in reflection port
        kappa_t:  cavity decay rate in transmission port
        kappa_loss: cavity loss rate
        gamma: emitter spontaneous decay rate
        C: cavity-emitter cooperativity
        gamma_dephasing: emitter dephasing rate (default 0)

    Returns:
        t: transmission coefficient
        r: reflection coefficient
        l: loss coefficient
    """
    kappa_tot = kappa_r + kappa_t + kappa_loss
    gamma_tot = gamma + gamma_dephasing

    C_eff = C  # /(1+2*delta_ac**2/kappa_tot**2)

    r = 1 - kappa_r / kappa_tot / (-1j * delta_cl / kappa_tot + 0.5 + C_eff / (-1j * delta_al / gamma_tot + 0.5))
    t = kappa_t / kappa_tot / (-1j * delta_cl / kappa_tot + 0.5 + C_eff / (-1j * delta_al / gamma_tot + 0.5))
    l = np.sqrt(1 - np.abs(t) ** 2 - np.abs(r) ** 2)

    return t, r, l


def cavity_qom_atom_centered(omega, delta, kappa_r, kappa_t, kappa_loss, gamma, C, gamma_dephasing=0):
    """
    Implementation of the cavity centered on atom frequency.

    Parameters:
        omega: probe laser frequency
        delta: detuning between emitter and cavity
        kappa_r:  cavity decay rate in reflection port
        kappa_t:  cavity decay rate in transmission port
        kappa_loss: cavity loss rate
        gamma: emitter spontaneous decay rate
        C: cavity-emitter cooperativity
        gamma_dephasing: emitter dephasing rate (default 0)

    Returns:
        t: transmission coefficient
        r: reflection coefficient
        l: loss coefficient
    """
    delta_ac = delta
    delta_cl = omega + delta
    delta_al = omega

    t, r, l = cavity_qom(
        delta_al,
        delta_ac,
        delta_cl,
        kappa_r,
        kappa_t,
        kappa_loss,
        gamma,
        C,
        gamma_dephasing,
    )
    return t, r, l


def cavity_qom_cavity_centered(omega, delta, kappa_r, kappa_t, kappa_loss, gamma, C, gamma_dephasing=0):
    """
    Implementation of the cavity centered on cavity frequency.

    Parameters:
        omega: probe laser frequency
        delta: detuning between emitter and cavity
        kappa_r:  cavity decay rate in reflection port
        kappa_t:  cavity decay rate in transmission port
        kappa_loss: cavity loss rate
        gamma: emitter spontaneous decay rate
        C: cavity-emitter cooperativity
        gamma_dephasing: emitter dephasing rate (default 0)

    Returns:
        t: transmission coefficient
        r: reflection coefficient
        l: loss coefficient
    """
    delta_ac = delta
    delta_cl = omega
    delta_al = delta + omega

    t, r, l = cavity_qom(
        delta_al,
        delta_ac,
        delta_cl,
        kappa_r,
        kappa_t,
        kappa_loss,
        gamma,
        C,
        gamma_dephasing,
    )
    return t, r, l


def cavity_qom_atom_centered_controlled(omega, delta, kappa_r, kappa_t, kappa_loss, gamma, C, N=0, gamma_dephasing=0):
    """
    Implementation of the cavity centered on cavity frequency.
    Cooperativity scales with number of atoms coupled.

    Parameters:
        omega: probe laser frequency
        delta: detuning between emitter and cavity
        kappa_r:  cavity decay rate in reflection port
        kappa_t:  cavity decay rate in transmission port
        kappa_loss: cavity loss rate
        gamma: emitter spontaneous decay rate
        C: cavity-emitter cooperativity
        N: number of atoms in the cavity (default 0)
        gamma_dephasing: emitter dephasing rate (default 0)

    Returns:
        t: transmission coefficient
        r: reflection coefficient
        l: loss coefficient
    """
    delta_ac = delta
    delta_cl = omega - delta
    delta_al = omega

    t, r, l = cavity_qom(
        delta_al,
        delta_ac,
        delta_cl,
        kappa_r,
        kappa_t,
        kappa_loss,
        gamma,
        N * C,
        gamma_dephasing,
    )
    return t, r, l


def cavity_enhanced_spontaneous_emission(kappa_in, kappa_loss, gamma, gamma_dephasing, DW, C):
    """
    Quantum optical modeling of cavity enhanced spontaneous emission.
    Formulas are reported in Appendix D of the reference article.

    Parameters:
        kappa_in: cavity decay rate in collected mode
        kappa_loss: cavity decay rate in non collected modes (loss)
        gamma: emitter spontaneous decay rate
        gamma_dephasing: emitter dephasing rate
        DW: Debye Waller factor (in general, branching ratio in the target optical transition compared to other transitions)
        C: cavity-emitter cooperativity

    Returns:
        p_coherent: probability of coherent spontaneous emission
        p_incoherent: probability of incoherent spontaneous emission
        p_2ph: probability of two-photon emission
        p_loss: probability of photon loss
    """
    gamma_r = gamma * DW
    p_coherent = (
        kappa_in
        / (kappa_in + kappa_loss)
        * C
        / (C + 1)
        * (C / DW)
        * gamma_r
        / (gamma + gamma_dephasing + (C / DW) * gamma_r)
    )
    p_incoherent = (
        kappa_in
        / (kappa_in + kappa_loss)
        * C
        / (C + 1)
        * gamma_dephasing
        / (gamma + gamma_dephasing + (C / DW) * gamma_r)
    )
    p_2ph = 0
    p_loss = 1 - p_coherent - p_incoherent - p_2ph
    return p_coherent, p_incoherent, p_2ph, p_loss
