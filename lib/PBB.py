import math

import numpy as np
import qutip as qt

import lib.NQobj as nq
import lib.states as st


def conditional_amplitude_reflection(r_u, t_u, l_u, r_d, t_d, l_d, dim=2):
    """
    Physical Building Block for conditional amplitude reflection
    (photon can be reflected, transmitted or lost depending on the spin state).
    Parameters:
        r_u, t_u, l_u: reflection, transmission and loss for Up spin state
        r_d, t_d, l_d: reflection, transmission and loss for Down spin state
        dim: dimension of the photon space (default 2)

    Returns:
        cavity operator
    """

    tot_d = np.abs(t_d) ** 2 + np.abs(r_d) ** 2 + np.abs(l_d) ** 2
    tot_u = np.abs(t_u) ** 2 + np.abs(r_u) ** 2 + np.abs(l_u) ** 2
    if not math.isclose(tot_d, 1, abs_tol=1e-6):
        raise ValueError("The squares of r_d, t_d and l_d should add up to 1.")
    if not math.isclose(tot_u, 1, abs_tol=1e-6):
        raise ValueError("The squares of r_u, t_u and l_u should add up to 1.")

    r = nq.name(qt.destroy(dim), "R")
    t = nq.name(qt.destroy(dim), "T")
    l = nq.name(qt.destroy(dim), "loss")
    u = nq.name(st.up, "spin")
    d = nq.name(st.down, "spin")

    L_u = np.abs(l_u) ** 2
    theta_loss_u = np.arctan(np.sqrt(L_u / (1 - L_u)))
    r_prime_u = r_u / np.sqrt(1 - L_u)
    t_prime_u = t_u / np.sqrt(1 - L_u)
    # t_prime_u, r_prime_u = r_prime_u, t_prime_u
    theta_splitting_u = np.arctan(t_prime_u / r_prime_u)
    cav_u = (
        (1j * np.angle(r_u) * r.dag() * r).expm()
        * (1j * np.angle(t_u) * t.dag() * t).expm()
        * (theta_splitting_u * (r.dag() * t - r * t.dag())).expm()
        * (theta_loss_u * (r.dag() * l - r * l.dag())).expm()
    )

    L_d = np.abs(l_d) ** 2
    theta_loss_d = np.arctan(np.sqrt(L_d / (1 - L_d)))
    r_prime_d = r_d / np.sqrt(1 - L_d)
    t_prime_d = t_d / np.sqrt(1 - L_d)
    # t_prime_d, r_prime_d = r_prime_d, t_prime_d
    theta_splitting_d = np.arctan(t_prime_d / r_prime_d)
    cav_d = (
        (1j * (np.angle(r_d) * r.dag() * r)).expm()
        * (1j * (np.angle(t_d) * t.dag() * t)).expm()
        * (theta_splitting_d * (r.dag() * t - r * t.dag())).expm()
        * (theta_loss_d * (r.dag() * l - r * l.dag())).expm()
    )
    cav_tot = nq.tensor(u.proj(), cav_u) + nq.tensor(d.proj(), cav_d)
    return cav_tot


def conditional_phase_reflection(r_u, l_u, r_d, l_d, dim=2):
    """
    Physical Building Block for conditional phase reflection
    (photon is either reflected or lost depending on the spin state).
    Parameters:
        r_u, l_u: reflection and loss for Up spin state
        r_d,  l_d: reflection and loss for Down spin state
        dim: dimension of the photon space (default 2)

    Returns:
        cavity operator
    """

    tot_d = np.abs(r_d) ** 2 + np.abs(l_d) ** 2
    tot_u = np.abs(r_u) ** 2 + np.abs(l_u) ** 2
    if not math.isclose(tot_d, 1, abs_tol=1e-6):
        raise ValueError("The squares of r_d and l_d should add up to 1.")
    if not math.isclose(tot_u, 1, abs_tol=1e-6):
        raise ValueError("The squares of r_u and l_u should add up to 1.")

    R = nq.name(qt.destroy(dim), "R")
    L = nq.name(qt.destroy(dim), "loss")
    u = nq.name(st.up, "spin")
    d = nq.name(st.down, "spin")

    theta_loss_u = np.arctan(np.abs(l_u) / np.abs(r_u))
    cav_u = (
        (1j * (np.angle(r_u) * R.dag() * R)).expm()
        * (1j * (np.angle(l_u) * L.dag() * L)).expm()
        * (theta_loss_u * (L.dag() * R - L * R.dag())).expm()
    )

    theta_loss_d = np.arctan(np.abs(l_d) / np.abs(r_d))
    cav_d = (
        (1j * (np.angle(r_d) * R.dag() * R)).expm()
        * (1j * (np.angle(l_d) * L.dag() * L)).expm()
        * (theta_loss_d * (L.dag() * R - L * R.dag())).expm()
    )

    return nq.tensor(u.proj(), cav_u) + nq.tensor(d.proj(), cav_d)


def unitary_beamsplitter(theta=0, dim=2):
    """
    Physical Building Block tha realizes a beam splitter operation

    Parameters
        theta: beam splitter rotation angle
    """
    A = nq.name(qt.destroy(dim), "A")
    B = nq.name(qt.destroy(dim), "B")
    return (theta * (A.dag() * B - A * B.dag())).expm()


def loss(loss=0.5, dim=2):
    """
    Physical Building Block tha realizes photon loss

    Parameters
        loss: loss factor
    """
    A = nq.name(qt.destroy(dim), "A", "oper")
    L = nq.name(qt.destroy(dim), "loss", "oper")
    theta = np.arctan(np.sqrt((loss) / (1 - loss)))
    return (theta * (A.dag() * L - A * L.dag())).expm()


def waveplate(theta=0, dim=2):
    """
    Physical Building Block tha realizes a rotation for polarization encoding.
    Uses default mode names H and V.

    Parameters
        theta: waveplate angle
    """
    r = nq.name(qt.destroy(dim), "H")
    l = nq.name(qt.destroy(dim), "V")

    return (-1j * theta * (r.dag() * l + r * l.dag())).expm()


def spontaneous_emission_ideal(dim=2):
    """
    Physical Building Block that realizes an ideal spin-dependent spontaneopus emission SPI.
    """
    photon = nq.name(qt.destroy(dim), "photon")
    I = nq.name(qt.qeye(dim), "photon", "oper")
    u = nq.name(st.up, "spin")
    d = nq.name(st.down, "spin")

    return nq.tensor(u.proj(), I) + nq.tensor(d.proj(), photon.dag())


def spontaneous_emission_error(dim=2):
    """
    Physical Building Block to model loss or errors (e.g. emission into extra modes) in spontaneopus emission SPI.
    """

    photon = nq.name(qt.destroy(dim), "photon")
    d = nq.name(st.down, "spin")

    return nq.tensor(d.proj(), photon.dag())


def spontaneous_two_photon_emission(dim=3):
    """
    Physical Building Block to model two-photon emission error in spontaneopus emission SPI.
    """

    photon = nq.name(qt.destroy(dim), "photon")
    d = nq.name(st.down, "spin")

    return nq.tensor(d.proj(), photon.dag() ** 2 / np.sqrt(2))


def phase(theta=0, dim=2):
    """
    Physical Building Block to model phase rotation in a photonic mode.
    """
    a = nq.name(qt.destroy(dim), "photon")
    return (1j * theta * a.dag() * a).expm()


def no_vacuum_projector(name, dim):
    """
    Physical Building Block to model the effect of detecting a photon, useful for heralding projectors.
    """
    identity = nq.name(qt.qeye(dim), name, "oper")
    vacuum = nq.name(st.vacuum_dm(dim), name, "oper")
    return identity - vacuum
