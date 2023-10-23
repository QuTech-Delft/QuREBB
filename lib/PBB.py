import math

import numpy as np
import qutip as qt

import lib.NQobj as nq
import lib.states as st


def conditional_amplitude_reflection(r_u, t_u, l_u, r_d, t_d, l_d, dim=2):
    """
    Physical Building Block for conditional amplitude reflection
    Depending on the spin state, the photon can be reflected, transmitted, or lost.

    Parameters:
        r_u, t_u, l_u : float
            reflection, transmission and loss for Up spin state
        r_d, t_d, l_d : float
            reflection, transmission and loss for Down spin state
        dim : int
            dimension of the photon space (default 2)

    Returns:
        cav_tot : NQobj
            A cavity operator combining the effect of reflection,
            transmission, and loss conditioned on the spin state.
    """

    # Check if the probabilities of reflection, transmission, and loss add up to 1 for both spin states.
    tot_d = np.abs(t_d) ** 2 + np.abs(r_d) ** 2 + np.abs(l_d) ** 2
    tot_u = np.abs(t_u) ** 2 + np.abs(r_u) ** 2 + np.abs(l_u) ** 2
    if not math.isclose(tot_d, 1, abs_tol=1e-6):
        raise ValueError("The squares of r_d, t_d and l_d should add up to 1.")
    if not math.isclose(tot_u, 1, abs_tol=1e-6):
        raise ValueError("The squares of r_u, t_u and l_u should add up to 1.")

    # Definitions of operators and states.
    r = nq.name(qt.destroy(dim), "R")
    t = nq.name(qt.destroy(dim), "T")
    l = nq.name(qt.destroy(dim), "loss")
    u = nq.name(st.up, "spin")
    d = nq.name(st.down, "spin")

    # Calculate the unitary transformations for the spin up state.
    L_u = np.abs(l_u) ** 2
    theta_loss_u = np.arctan(np.sqrt(L_u / (1 - L_u)))
    r_prime_u = r_u / np.sqrt(1 - L_u)
    t_prime_u = t_u / np.sqrt(1 - L_u)

    theta_splitting_u = np.arctan(np.abs(t_prime_u) / np.abs(r_prime_u))
    cav_u = (
        (1j * np.angle(r_u) * r.dag() * r).expm()
        * (1j * np.angle(t_u) * t.dag() * t).expm()
        * (theta_splitting_u * (r.dag() * t - r * t.dag())).expm()
        * (theta_loss_u * (r.dag() * l - r * l.dag())).expm()
    )

    # Calculate the unitary transformations for the spin down state.
    L_d = np.abs(l_d) ** 2
    theta_loss_d = np.arctan(np.sqrt(L_d / (1 - L_d)))
    r_prime_d = r_d / np.sqrt(1 - L_d)
    t_prime_d = t_d / np.sqrt(1 - L_d)

    theta_splitting_d = np.arctan(np.abs(t_prime_d) / np.abs(r_prime_d))
    cav_d = (
        (1j * (np.angle(r_d) * r.dag() * r)).expm()
        * (1j * (np.angle(t_d) * t.dag() * t)).expm()
        * (theta_splitting_d * (r.dag() * t - r * t.dag())).expm()
        * (theta_loss_d * (r.dag() * l - r * l.dag())).expm()
    )

    # Combine the unitary transformations for both spin states.
    cav_tot = nq.tensor(u.proj(), cav_u) + nq.tensor(d.proj(), cav_d)

    return cav_tot


def conditional_phase_reflection(r_u, l_u, r_d, l_d, dim=2):
    """
    Physical Building Block for conditional phase reflection
    Depending on the spin state, the photon can be reflected or lost.
    Parameters:
        r_u, l_u : float
            reflection and loss for Up spin state
        r_d, l_d : float
            reflection and loss for Down spin state
        dim : int
            dimension of the photon space (default 2)

    Returns:
        cav_tot : NQobj
            A cavity operator combining the effect of reflection and loss
            conditioned on the spin state.
    """

    # Check if the probabilities of reflection, transmission, and loss add up to 1 for both spin states.
    tot_d = np.abs(r_d) ** 2 + np.abs(l_d) ** 2
    tot_u = np.abs(r_u) ** 2 + np.abs(l_u) ** 2
    if not math.isclose(tot_d, 1, abs_tol=1e-6):
        raise ValueError("The squares of r_d and l_d should add up to 1.")
    if not math.isclose(tot_u, 1, abs_tol=1e-6):
        raise ValueError("The squares of r_u and l_u should add up to 1.")

    # Definitions of operators and states.
    R = nq.name(qt.destroy(dim), "R")
    L = nq.name(qt.destroy(dim), "loss")
    u = nq.name(st.up, "spin")
    d = nq.name(st.down, "spin")

    # Calculate the unitary transformations for the spin up state.
    theta_loss_u = np.arctan(np.abs(l_u) / np.abs(r_u))
    cav_u = (
        (1j * (np.angle(r_u) * R.dag() * R)).expm()
        * (1j * (np.angle(l_u) * L.dag() * L)).expm()
        * (theta_loss_u * (L.dag() * R - L * R.dag())).expm()
    )

    # Calculate the unitary transformations for the spin down state.
    theta_loss_d = np.arctan(np.abs(l_d) / np.abs(r_d))
    cav_d = (
        (1j * (np.angle(r_d) * R.dag() * R)).expm()
        * (1j * (np.angle(l_d) * L.dag() * L)).expm()
        * (theta_loss_d * (L.dag() * R - L * R.dag())).expm()
    )

    # Combine the unitary transformations for both spin states.
    cav_tot = nq.tensor(u.proj(), cav_u) + nq.tensor(d.proj(), cav_d)
    return cav_tot


def unitary_beamsplitter(theta=0, dim=2):
    """
    Physical Building Block of a beam splitter.
    The parameter `theta` determines the proportion of split between the two paths.

    Parameters:
        theta : float
            Beam splitter rotation angle. Determines the splitting ratio.
            0      -> perfect transmission
            pi/4   -> split equally to reflection and transmission.
        dim : int
            Dimension of the photonic mode, default is 2 (vacuum and single-photon state).

    Returns:
        BS : NQobj
            Beam splitter operator
    """

    A = nq.name(qt.destroy(dim), "A")
    B = nq.name(qt.destroy(dim), "B")

    BS = (theta * (A.dag() * B - A * B.dag())).expm()

    return BS


def loss(loss=0.5, dim=2):
    """
    Physical Building Block for photon loss.
    The parameter `loss` is the probability of photon loss.

    Parameters:
        loss : float
            Loss factor representing the probability of photon loss.
            0   -> no loss
            0.5 -> 50% chance of loss
            1   -> complete loss
        dim : int
            Dimension of the photonic mode, default is 2 (vacuum and single-photon state).

    Returns:
        LossOp : NQobj
            Photon loss operator.
    """

    A = nq.name(qt.destroy(dim), "A", "oper")
    L = nq.name(qt.destroy(dim), "loss", "oper")

    theta = np.arctan(np.sqrt((loss) / (1 - loss)))

    LossOp = (theta * (A.dag() * L - A * L.dag())).expm()

    return LossOp


def waveplate(theta=0, dim=2):
    """
    Physical Building Block of a waveplate.
    Change the polarization-encoded photon statea.
    Uses H and V as default names.

    Parameters:
        theta : float
            Waveplate angle.
            0       -> no change in polarization
            pi/4    -> rotates the polarization by 45 degrees
        dim : int
            Dimension of the photonic mode, default is 2 (vacuum and single-photon state).

    Returns:
        WP : NQobj
            Waveplate operator.
    """

    r = nq.name(qt.destroy(dim), "H")
    l = nq.name(qt.destroy(dim), "V")

    WP = (-1j * theta * (r.dag() * l + r * l.dag())).expm()

    return WP


def spontaneous_emission_ideal(dim=2):
    """
    Physical Building Block of an ideal spin-dependent spontaneous emission (SPI).
    Ideal case where the emission is perfectly spin-dependent.
    Down is the bright state, Up is the dark state

    Parameters:
        dim : int
            Dimension of the photonic mode, default is 2 (vacuum and single-photon state).

    Returns:
        SE_ideal : NQobj
            Operator for ideal spin-dependent spontaneous emission.
    """

    photon = nq.name(qt.destroy(dim), "photon")
    I = nq.name(qt.qeye(dim), "photon", "oper")
    u = nq.name(st.up, "spin")
    d = nq.name(st.down, "spin")

    SE_ideal = nq.tensor(u.proj(), I) + nq.tensor(d.proj(), photon.dag())

    return SE_ideal


def spontaneous_emission_error(dim=2):
    """
    Physical Building Block of an erroneous spin-dependent spontaneous emission (SPI).
    Models emission errors like loss or unintended emissions into extra modes.
    Down is the bright state, and emission in this context signifies an error.

    Parameters:
        dim : int
            Dimension of the photonic mode, default is 2 (vacuum and single-photon state).

    Returns:
        SE_error : NQobj
            Operator for erroneous spin-dependent spontaneous emission.
    """

    photon = nq.name(qt.destroy(dim), "photon")
    d = nq.name(st.down, "spin")

    SE_error = nq.tensor(d.proj(), photon.dag())

    return SE_error


def spontaneous_two_photon_emission(dim=3):
    """
    Physical Building Block of a two-photon emission error in spin-dependent spontaneous emission (SPI).
    Models the case where two photons are emitted simultaneously.
    Down is the bright state.

    Parameters:
        dim : int
            Dimension of the photonic mode, default is 3 (vacuum, single-photon, and two-photon states).

    Returns:
        SE_two_photon : NQobj
            Operator for two-photon emission error in spin-dependent spontaneous emission.
    """

    photon = nq.name(qt.destroy(dim), "photon")
    d = nq.name(st.down, "spin")

    SE_two_photon = nq.tensor(d.proj(), photon.dag() ** 2 / np.sqrt(2))

    return SE_two_photon


def phase(theta=0, dim=2):
    """
    Physical Building Block of a phase shift of a photonic mode.

    Parameters:
        theta : float
            Phase shift.
        dim : int
            Dimension of the photonic mode, default is 2 (vacuum and single-photon state).

    Returns:
        phase_operator : NQobj
            Phase shift operator.
    """

    a = nq.name(qt.destroy(dim), "photon")

    phase_operator = (1j * theta * a.dag() * a).expm()

    return phase_operator


def no_vacuum_projector(name, dim):
    """
    Physical Building Block for a no-vacuum projector.
    Projects state except the vacuum state, ensuring the presence of a photon.
    Useful for heralding (photodetection).

    Parameters:
        name : str
            Name of the photonic mode.
        dim : int
            Dimension of the photonic mode.

    Returns:
        no_vacuum : NQobj
            Proejctor excluding the vacuum state.
    """
    identity = nq.name(qt.qeye(dim), name, "oper")
    vacuum = nq.name(st.vacuum_dm(dim), name, "oper")

    no_vacuum = identity - vacuum
    return no_vacuum
