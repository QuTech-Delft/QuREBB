import numpy as np
import qutip as qt

import lib.NQobj as nq
import lib.PBB as pbb
import lib.quantum_optical_modelling as qom
import lib.states as st

# Covenience function for tracing out


def trace_out_loss_modes(Q):
    loss_modes = [x for x in Q.names[0] if "loss" in x]
    return Q.ptrace(loss_modes, keep=False)


def trace_out_everything_but_spins(Q):
    classic_spin_names = ["spin", "Alice", "Bob", "Charlie", "alice", "bob", "charlie"]
    spin_modes = [x for x in Q.names[0] if x in classic_spin_names]
    return Q.ptrace(spin_modes)


###########
## SPIs  ##
###########


def SpontaneousEmissionFockSPI(
    dm_in, spin_name, photon_name, dim, kappa_in, kappa_loss, gamma, g, DW, QE, gamma_dephasing=0, ideal=False, **kw
):

    C = 4 * g**2 / (kappa_in + kappa_loss) / (gamma + gamma_dephasing)

    if ideal:
        p_coh, p_incoh, p_2ph, p_loss = 1, 0, 0, 0
    else:
        p_coh, p_incoh, p_2ph, p_loss = qom.cavity_enhanced_spontaneous_emission(
            kappa_in, kappa_loss, gamma, gamma_dephasing, DW * QE, C
        )

    c_coh = pbb.spontaneous_emission_ideal(dim=dim)
    c_coh.rename("spin", spin_name)
    c_coh.rename("photon", photon_name)

    c_loss = pbb.spontaneous_emission_error(dim=dim)
    c_loss.rename("spin", spin_name)
    c_loss.rename("photon", "loss")

    c_incoh = pbb.spontaneous_emission_error(dim=dim)
    c_incoh.rename("spin", spin_name)
    c_incoh.rename("photon", f"{photon_name}_incoh")

    c_2ph = pbb.spontaneous_two_photon_emission(dim=dim)
    c_2ph.rename("spin", spin_name)
    c_2ph.rename("photon", photon_name)
    c = np.sqrt(p_coh) * c_coh + np.sqrt(p_loss) * c_loss
    dm_out = c * dm_in * c.dag() + p_incoh * c_incoh * dm_in * c_incoh.dag() + p_2ph * c_2ph * dm_in * c_2ph.dag()

    return trace_out_loss_modes(dm_out)


def ConditionalAmplitudeReflectionTimeBinSPI(
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
    **kwargs,
):

    # cavity params must be [f, kappa_in, kappa_loss, gamma, delta, splitting, g]
    if ideal:
        t_u = r_d = 1
        t_d = r_u = 0
        l_u = l_d = 0
    else:
        C = 4 * g**2 / (kappa_t + kappa_r + kappa_loss) / (gamma + gamma_dephasing)

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

    cav = pbb.conditional_amplitude_reflection(r_u, t_u, l_u, r_d, t_d, l_d, dim=dim)
    cav.rename("spin", spin_name)

    # Apply early
    cav.rename("R", photon_early_name)
    cav.rename("T", "loss_transmission")
    cav.rename("loss", "loss")
    dm_E_full = cav * dm_in * cav.dag()
    dm_E = trace_out_loss_modes(dm_E_full)

    # Flip spin and scatter late
    RX_pi = nq.NQobj([[0, 1], [1, 0]], names=spin_name, kind="oper")
    cav.rename(photon_early_name, photon_late_name)
    dm_L_full = cav * (RX_pi * dm_E * RX_pi.dag()) * cav.dag()
    dm_L = trace_out_loss_modes(dm_L_full)

    return dm_L


#######################
## Photon operations ##
#######################


def HOM(dm_in, photon_names, dim=3, **kwargs):

    hom_bs = pbb.unitary_beamsplitter(theta=np.pi / 4, dim=dim)
    hom_bs.rename("A", photon_names[0])
    hom_bs.rename("B", photon_names[1])

    return hom_bs * dm_in * hom_bs.dag()


def BasisRotation(dm_in, photon_names=None, sign=+1, dim=2, **kwargs):
    wp = pbb.unitary_beamsplitter(sign * np.pi / 4, dim=dim)
    wp.rename("A", photon_names[0])
    wp.rename("B", photon_names[1])

    return wp * dm_in * wp.dag()


def ModeLoss(dm_in, photon_name, loss, dim, ideal=False, **kwargs):
    if ideal:
        loss = 0
    link_loss = pbb.loss(loss, dim=dim)
    link_loss.rename("A", photon_name)
    dm_loss = link_loss * dm_in * link_loss.dag()

    dm_out = dm_loss.ptrace(["loss"], keep=False)
    return dm_out


###############
##  Sources  ##
###############


def PhotonSourceTimeBin(dm_in, photon_early_name, photon_late_name, dim, alpha=None, **kw):
    """alpha is a bool and not none as this cant be saved as the attrs of a xarray dataset."""
    if alpha is None:
        photon_basis = st.photon(dim)
    else:
        photon_basis = qt.coherent(dim, alpha)
    E = nq.name(photon_basis, photon_early_name, "state")
    L = nq.name(photon_basis, photon_late_name, "state")
    return nq.tensor(dm_in, nq.ket2dm((E + L).unit()))


#################
##  Spin gates ##
#################


def SpinPiX(dm_in, spin_name, **kwargs):
    RX_pi = nq.name(qt.sigmax(), names=spin_name)
    return RX_pi * dm_in * RX_pi.dag()


def SpinPiY(dm_in, spin_name, **kwargs):
    RY_pi = nq.name(qt.sigmay(), names=spin_name)
    return RY_pi * dm_in * RY_pi.dag()


########################
##  Detect & Measure  ##
########################


def Herald(dm_in, herald_projector, **kw):
    dm_final = herald_projector * dm_in * herald_projector.dag()
    return trace_out_everything_but_spins(dm_final)


############################
##  Noise & Imperfections ##
############################


def DarkCounts(dm_in, photon_name, dc_rate, dim, ideal=False, **kwargs):

    if ideal:
        dc_rate = 0
    a = nq.name(qt.destroy(dim), photon_name)

    return (dc_rate) * (a.dag() * dm_in * a) + (1 - dc_rate) * dm_in
