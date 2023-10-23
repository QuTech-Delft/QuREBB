import numpy as np
import qutip as qt

import lib.NQobj as nq
import lib.PBB as pbb
import lib.quantum_optical_modelling as qom
import lib.states as st

# Covenience function for tracing out


def trace_out_loss_modes(Q):

    """
    Trace out any modes from the quantum object Q that have the term 'loss' in their name.

    Parameters:
    -----------
    Q : NQobj
        The quantum object whose modes need to be traced out.

    Returns:
    --------
    NQobj
        Quantum object after tracing out the loss modes.
    """

    loss_modes = [x for x in Q.names[0] if "loss" in x]
    return Q.ptrace(loss_modes, keep=False)


def trace_out_everything_but_spins(Q):

    """
    Trace out all modes from the quantum object Q except for the ones with classic spin names.

    Parameters:
    -----------
    Q : NQobj
        The quantum object whose modes need to be traced out.

    Returns:
    --------
    NQobj
        Quantum object after tracing out all modes except spins.
    """

    classic_spin_names = ["Spin", "spin", "Alice", "Bob", "Charlie", "alice", "bob", "charlie"]
    spin_modes = [x for x in Q.names[0] if x in classic_spin_names]
    return Q.ptrace(spin_modes)


####################################
##  Spin-Photon Interfaces (SPI)  ##
####################################


def spontaneous_emission_fock_spi(
    dm_in, spin_name, photon_name, dim, kappa_in, kappa_loss, gamma, g, DW, QE, gamma_dephasing=0, ideal=False, **kw
):

    """
    Simulate the spontaneous emission process for a spin and Fock-state encoded photonic mode.

    Parameters:
    -----------
    dm_in : NQobj
        Initial density matrix of the quantum system.
    spin_name : str
        Choice of the spin.
    photon_name : str
        Choice of the photonic mode.
    dim : int
        Dimension of the photonic mode (e.g. dim = 2 for |n=0> and |n=1> (vacuum and single photon)).
    kappa_in, kappa_loss, gamma, g, DW, QE : float
        Parameters characterizing a cQED system.
    gamma_dephasing : float, optional
        Rate of dephasing. Default is 0.
    ideal : bool, optional
        If True, simulates the ideal spontaneous emission.
        Default is False.

    Returns:
    --------
    NQobj
        Output density matrix after the spontaneous emission.
    """

    # If the process is ideal, set the probabilities accordingly
    if ideal:
        p_coh, p_incoh, p_2ph, p_loss = 1, 0, 0, 0
    else:
        C = 4 * g**2 / (kappa_in + kappa_loss) / (gamma + gamma_dephasing)
        p_coh, p_incoh, p_2ph, p_loss = qom.cavity_enhanced_spontaneous_emission(
            kappa_in, kappa_loss, gamma, gamma_dephasing, DW * QE, C
        )

    # Define the coherent channel
    c_coh = pbb.spontaneous_emission_ideal(dim=dim)
    c_coh.rename("spin", spin_name)
    c_coh.rename("photon", photon_name)

    # Define the loss channel
    c_loss = pbb.spontaneous_emission_error(dim=dim)
    c_loss.rename("spin", spin_name)
    c_loss.rename("photon", "loss")

    # Define the incoherent channel
    c_incoh = pbb.spontaneous_emission_error(dim=dim)
    c_incoh.rename("spin", spin_name)
    c_incoh.rename("photon", f"{photon_name}_incoh")

    # Define the two-photon emission channel
    c_2ph = pbb.spontaneous_two_photon_emission(dim=dim)
    c_2ph.rename("spin", spin_name)
    c_2ph.rename("photon", photon_name)

    # Compute the output density matrix using the defined channels
    c = np.sqrt(p_coh) * c_coh + np.sqrt(p_loss) * c_loss
    dm_out = c * dm_in * c.dag() + p_incoh * c_incoh * dm_in * c_incoh.dag() + p_2ph * c_2ph * dm_in * c_2ph.dag()

    return trace_out_loss_modes(dm_out)


def conditional_amplitude_reflection_time_bin_spi(
    dm_in,
    spin_name,
    photon_early_name,
    photon_late_name,
    dim,
    atom_centered=True,
    f_operation=None,
    kappa_r=None,
    kappa_t=None,
    kappa_loss=0,
    gamma=None,
    delta=None,
    splitting=None,
    g=None,
    ideal=False,
    gamma_dephasing=0,
    **kw,
):

    """
    Simulate the conditional amplitude reflection with a time-bin encoded photonic mode.

    Parameters:
    -----------
    dm_in : NQobj
        Initial density matrix of the quantum system.
    spin_name, photon_early_name, photon_late_name : str
        Names for the spin mode and the early and late photon modes.
    dim : int
        Dimension of the photonic mode (e.g. dim = 2 for |n=0> and |n=1> (vacuum and single photon)).
    atom_centered : bool, optional
        Whether the frequency of photonic mode are centered around the atom or the cavity.
        Default is True.
    f_operation :
        frequency of photonic mode
    kappa_r, kappa_t, kappa_loss, gamma, delta, splitting, g, gamma_dephasing : float
        Parameters characterizing a cQED system.
    ideal : bool, optional
        If True, simulates the ideal reflection.
        Default is False.

    Returns:
    --------
    NQobj
        Output density matrix after the conditional amplitude reflection process.

    Description:
    ------------
    - Ideal case:
      perfect reflection without any transmission or losses.
      (the reflection coefficient is one and the others are zero)

    - Realistic case:
      Based on the provided cQED parameters and the choice of centeredness (atom or cavity),
      the function calculates the reflection, transmission, and loss coefficients.

    It then performs the conditional amplitude reflection operation on the input density matrix and
    returns the processed output density matrix.
    """

    # If ideal, directly set reflection and transmission probabilities
    if ideal:
        t_u = r_d = 1
        t_d = r_u = 0
        l_u = l_d = 0

    # If not ideal, check if all required parameters are provided
    else:
        if None in (f_operation, kappa_r, kappa_t, gamma, delta, splitting, g):
            raise ValueError(
                "In not ideal then f_operation, kappa_r, kappa_t, gamma, delta, splitting, g should all be defined."
            )

        C = 4 * g**2 / (kappa_t + kappa_r + kappa_loss) / (gamma + gamma_dephasing)

        # Calculate the conditional amplitude reflection based on centeredness and provided parameters
        if atom_centered:
            t_u, r_u, l_u = qom.cavity_qom_atom_centered(
                f_operation, -delta, kappa_r, kappa_t, kappa_loss, gamma, C, gamma_dephasing=gamma_dephasing
            )
            t_d, r_d, l_d = qom.cavity_qom_atom_centered(
                f_operation + splitting / 2,
                -delta - splitting / 2,
                kappa_r,
                kappa_t,
                kappa_loss,
                gamma,
                C,
                gamma_dephasing=gamma_dephasing,
            )

        else:
            t_u, r_u, l_u = qom.cavity_qom_cavity_centered(
                f_operation, delta, kappa_r, kappa_t, kappa_loss, gamma, C, gamma_dephasing=gamma_dephasing
            )
            t_d, r_d, l_d = qom.cavity_qom_cavity_centered(
                f_operation, delta - splitting, kappa_r, kappa_t, kappa_loss, gamma, C, gamma_dephasing=gamma_dephasing
            )

    # Implement the conditional amplitude reflection operation with the physical building block (PBB)
    cav = pbb.conditional_amplitude_reflection(r_u, t_u, l_u, r_d, t_d, l_d, dim=dim)
    cav.rename("spin", spin_name)

    # The early reflection process and obtain the resulting density matrix
    cav.rename("R", photon_early_name)
    cav.rename("T", "loss_transmission")
    cav.rename("loss", "loss")

    dm_E_full = cav * dm_in * cav.dag()
    dm_E = trace_out_loss_modes(dm_E_full)

    # Flip the spin state and the late reflection process
    RX_pi = nq.NQobj([[0, 1], [1, 0]], names=spin_name, kind="oper")
    cav.rename(photon_early_name, photon_late_name)
    dm_L_full = cav * (RX_pi * dm_E * RX_pi.dag()) * cav.dag()
    dm_L = trace_out_loss_modes(dm_L_full)

    return dm_L


###########################
##  Photonic operations  ##
###########################


def hom(dm_in, photon_names, dim, **kw):
    """
    Simulate the Hong-Ou-Mandel (HOM) interference with a 50:50 beamsplitter.

    Parameters:
    -----------
    dm_in : NQobj
        Initial density matrix of the spin + photonic modes.
    photon_names : list of str
        Names assigned to the two photonic modes to be interfered.
    dim : int
        Dimension of the photonic mode (e.g. dim = 3 for vacuum, single-photon, and two-photon states).

    Returns:
    --------
    NQobj
        Output density matrix of teh whole quantum system after the HOM interference.
    """

    # Create a unitary beamsplitter operation with the beamsplitter physical building block (PBB)
    hom_bs = pbb.unitary_beamsplitter(theta=np.pi / 4, dim=dim)
    # Rename the first output mode of the beamsplitter with the first photon name provided
    hom_bs.rename("A", photon_names[0])
    # Rename the second output mode of the beamsplitter with the second photon name provided
    hom_bs.rename("B", photon_names[1])

    return hom_bs * dm_in * hom_bs.dag()


def basis_rotation(dm_in, photon_names, dim, sign=+1, **kw):

    """
    Perform a basis rotation on the photonic system using a beamsplitter.

    Parameters:
    -----------
    dm_in : NQobj
        Initial density matrix of the whole quantum system.
    photon_names : list of str
        Names assigned to the two photonic input modes.
    dim : int
        Dimension of the photonic mode (e.g. dim = 2 for |n=0> and |n=1> (vacuum and single photon)).
    sign : int, optional
        Sign for the rotation (either +1 or -1). Default is +1.

    Returns:
    --------
    NQobj
        Output density matrix of the whole quantum system after the basis rotation.
    """

    wp = pbb.unitary_beamsplitter(sign * np.pi / 4, dim=dim)
    wp.rename("A", photon_names[0])
    wp.rename("B", photon_names[1])

    return wp * dm_in * wp.dag()


def mode_loss(dm_in, photon_name, loss, dim, ideal=False, **kw):
    """
    Simulate loss of a photonic mode.

    Parameters:
    -----------
    dm_in : NQobj
        Initial density matrix of the whole quantum system.
    photon_name : str
        Names assigned to the photonic input modes.
    loss : float
        The loss between 0 to 1. (0: no loss, 1: complete loss)
    dim : int
        Dimension of the photonic mode (e.g. dim = 2 for |n=0> and |n=1> (vacuum and single photon)).
    ideal : bool, optional
        If True, loss = 0.
        Default is False.

    Returns:
    --------
    NQobj
        Output density matrix of the whole quantum system after photonic loss.
    """

    if ideal:
        loss = 0

    link_loss = pbb.loss(loss, dim=dim)
    link_loss.rename("A", photon_name)

    dm_loss = link_loss * dm_in * link_loss.dag()
    dm_out = dm_loss.ptrace(["loss"], keep=False)

    return dm_out


######################
##  Photon Sources  ##
######################


def photon_source_time_bin(dm_in, photon_early_name, photon_late_name, dim, alpha=None, **kw):
    """
    Generate a time-bin encoded photonic qubit.

    Parameters:
    ----------
    dm_in : NQobj
        Density matrix that time-bin encoded photon will be added to .
    photon_early_name, photon_late_name : str
        Names of the early and late photon modes.
    dim : int
        Dimension of the photonic mode (e.g. dim = 2 for |n=0> and |n=1> (vacuum and single photon)).
    alpha : complex or None, optional
        If specified, it represents the coherent state amplitude.
        Otherwise, a single photon state is assumed.

    Returns:
    --------
    NQobj
        Resulting density matrix after generating the time-bin encoded photon.
    """

    # If alpha is not provided, assume a single photon state
    if alpha is None:
        photon_basis = st.photon(dim)

    # Else, generate a coherent state with the given amplitude
    else:
        photon_basis = qt.coherent(dim, alpha)

    # Name the early- and late-time bin modes
    E = nq.name(photon_basis, photon_early_name, "state")
    L = nq.name(photon_basis, photon_late_name, "state")

    # Return the tensor product of the input density matrix with the time-bin encoded state
    return nq.tensor(dm_in, nq.ket2dm((E + L).unit()))


##################
##  Spin gates  ##
##################


def spin_pi_x(dm_in, spin_name, **kw):
    """
    Apply a pi rotation aroud x-axis.

    Parameters:
    ----------
    dm_in : NQobj
        Initial density matrix of the whole quantum system.
    spin_name : str
        Name of the spin part to be rotated.

    Returns:
    --------
    NQobj
        Output density matrix after applying the X gate (upto global phase).
    """

    # Define the pi rotation operator about x-axis
    RX_pi = nq.name(qt.sigmax(), names=spin_name)
    return RX_pi * dm_in * RX_pi.dag()


def spin_pi_y(dm_in, spin_name, **kw):
    """
    Apply a pi rotation aroud y-axis.

    Parameters:
    ----------
    dm_in : NQobj
        Input density matrix of the whole quantum system.
    spin_name : str
        Name of the spin part to be rotated.

    Returns:
    --------
    NQobj
        Output density matrix after applying the Y gate (upto global phase).
    """

    # Define the pi rotation operator about y-axis
    RY_pi = nq.name(qt.sigmay(), names=spin_name)
    return RY_pi * dm_in * RY_pi.dag()


########################
##  Detect & Measure  ##
########################


def herald(dm_in, herald_projector, **kw):
    """
    Heralding with the given projector operator

    Parameters:
    ----------
    dm_in : NQobj
        Input density matrix of the whole quantum system.
    herald_projector : NQobj
        Projector operator representing heralding.

    Returns:
    --------
    NQobj
        Output density matrix after heralding.
    """

    # Apply the heralding to the density matrix
    dm_final = herald_projector * dm_in * herald_projector.dag()
    return trace_out_everything_but_spins(dm_final)


#############################
##  Noise & Imperfections  ##
#############################


def dark_counts(dm_in, photon_name, dc_rate, dim, ideal=False, **kw):
    """
    Dark counts.

    Parameters:
    ----------
    dm_in : NQobj
        Input density matrix of the whole quantum system.
    photon_name : str
        Photonic mode that the dark count will be added to.
    dc_rate : float
        Rate of dark counts. (Probability)
    dim : int
        Dimension of the photonic mode (e.g. dim = 2 for |n=0> and |n=1> (vacuum and single photon)).
    ideal : bool, optional
        If True, the dark count rate is set to zero.
        Default is False.

    Returns:
    --------
    NQobj
        Output density matrix after the dark counts.
    """

    if ideal:
        dc_rate = 0
    a = nq.name(qt.destroy(dim), photon_name)

    # Photon is added to the designated mode.
    return (dc_rate) * (a.dag() * dm_in * a) + (1 - dc_rate) * dm_in
