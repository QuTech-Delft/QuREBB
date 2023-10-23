import numpy as np

import lib.LBB as lbb
import lib.NQobj as nq
import lib.states as st
from lib.PBB import no_vacuum_projector
from lib.protocol import Protocol


class ProtocolA(Protocol):
    def __init__(self, parameters: dict):
        super().__init__(parameters)

        self.dm_init = nq.tensor(
            nq.name(st.alpha_dm(self.parameters["alpha"]), "Alice", "state"),
            nq.name(st.alpha_dm(self.parameters["alpha"]), "Bob", "state"),
        )

        self.herald_projectors = [
            (
                no_vacuum_projector("Pa", self.dim)
                + no_vacuum_projector("Pa_incoh", self.dim)
                + no_vacuum_projector("Pa_incoh2", self.dim)
            ),
            (
                no_vacuum_projector("Pb", self.dim)
                + no_vacuum_projector("Pb_incoh", self.dim)
                + no_vacuum_projector("Pb_incoh2", self.dim)
            ),
        ]

        state_ud = nq.tensor(nq.name(st.up, "Alice", "state"), nq.name(st.down, "Bob", "state"))
        state_du = nq.tensor(nq.name(st.down, "Alice", "state"), nq.name(st.up, "Bob", "state"))
        self.target_states = [(state_ud + state_du).unit(), (state_ud - state_du).unit()]

    def protocol_sequence(self):
        self.do_lbb(lbb.spontaneous_emission_fock_spi, spin_name="Alice", photon_name="Pa")
        self.do_lbb_on_photons(lbb.mode_loss, photon_names=["Pa", "Pa_incoh"], loss=self.parameters["insertion_loss"])

        self.do_lbb(lbb.spontaneous_emission_fock_spi, spin_name="Bob", photon_name="Pb")
        self.do_lbb_on_photons(lbb.mode_loss, photon_names=["Pb", "Pb_incoh"], loss=self.parameters["insertion_loss"])

        self.do_lbb_on_photons(
            lbb.mode_loss,
            photon_names=["Pa", "Pa_incoh", "Pb", "Pb_incoh"],
            loss=1 - np.sqrt(1 - self.parameters["link_loss"]),
        )

        self.do_lbb(lbb.hom, photon_names=["Pa", "Pb"])
        self.do_lbb(lbb.hom, photon_names=["Pa_incoh", "Pb_incoh2"])
        self.do_lbb(lbb.hom, photon_names=["Pb_incoh", "Pa_incoh2"])


class ProtocolB(Protocol):
    def __init__(self, parameters: dict):
        super().__init__(parameters)

        self.dm_init = nq.tensor(nq.name(st.x_dm, "Alice", "state"), nq.name(st.x_dm, "Bob", "state"))

        self.herald_projectors = [
            (
                nq.tensor(no_vacuum_projector("Ea", self.dim), no_vacuum_projector("La", self.dim))
                + nq.tensor(no_vacuum_projector("Eb", self.dim), no_vacuum_projector("Lb", self.dim))
            ),
            (
                nq.tensor(no_vacuum_projector("Ea", self.dim), no_vacuum_projector("Lb", self.dim))
                + nq.tensor(no_vacuum_projector("Eb", self.dim), no_vacuum_projector("La", self.dim))
            ),
        ]

        state_ud = nq.tensor(nq.name(st.up, "Alice", "state"), nq.name(st.down, "Bob", "state"))
        state_du = nq.tensor(nq.name(st.down, "Alice", "state"), nq.name(st.up, "Bob", "state"))
        self.target_states = [(state_ud + state_du).unit(), (state_ud - state_du).unit()]

    def protocol_sequence(self):
        self.do_lbb(lbb.photon_source_time_bin, photon_early_name="Ea", photon_late_name="La")
        self.do_lbb_on_photons(lbb.mode_loss, photon_names=["Ea", "La"], loss=self.parameters["insertion_loss"])
        self.do_lbb(
            lbb.conditional_amplitude_reflection_time_bin_spi,
            photon_early_name="Ea",
            photon_late_name="La",
            spin_name="Alice",
        )
        self.do_lbb_on_photons(lbb.mode_loss, photon_names=["Ea", "La"], loss=self.parameters["insertion_loss"])

        self.do_lbb(lbb.photon_source_time_bin, photon_early_name="Eb", photon_late_name="Lb")
        self.do_lbb_on_photons(lbb.mode_loss, photon_names=["Eb", "Lb"], loss=self.parameters["insertion_loss"])
        self.do_lbb(
            lbb.conditional_amplitude_reflection_time_bin_spi,
            photon_early_name="Eb",
            photon_late_name="Lb",
            spin_name="Bob",
        )
        self.do_lbb_on_photons(lbb.mode_loss, photon_names=["Eb", "Lb"], loss=self.parameters["insertion_loss"])

        self.do_lbb_on_photons(
            lbb.mode_loss, photon_names=["Ea", "Eb", "La", "Lb"], loss=1 - np.sqrt(1 - self.parameters["link_loss"])
        )

        self.do_lbb(lbb.hom, photon_names=["Ea", "Eb"])
        self.do_lbb(lbb.hom, photon_names=["La", "Lb"])


class ProtocolC(Protocol):
    def __init__(self, parameters: dict):
        super().__init__(parameters)

        self.dm_init = nq.tensor(nq.name(st.x_dm, "Alice", "state"), nq.name(st.x_dm, "Bob", "state"))

        self.herald_projectors = [
            no_vacuum_projector(name="E", dim=self.dim),
            no_vacuum_projector(name="L", dim=self.dim),
        ]

        state_uu = nq.tensor(nq.name(st.up, "Alice", "state"), nq.name(st.up, "Bob", "state"))
        state_dd = nq.tensor(nq.name(st.down, "Alice", "state"), nq.name(st.down, "Bob", "state"))
        self.target_states = [(state_uu + state_dd).unit(), (state_uu - state_dd).unit()]

    def protocol_sequence(self):
        self.do_lbb(lbb.photon_source_time_bin, photon_early_name="E", photon_late_name="L")

        self.do_lbb_on_photons(lbb.mode_loss, photon_names=["E", "L"], loss=self.parameters["insertion_loss"])
        self.do_lbb(
            lbb.conditional_amplitude_reflection_time_bin_spi,
            photon_early_name="E",
            photon_late_name="L",
            spin_name="Alice",
        )
        self.do_lbb_on_photons(lbb.mode_loss, photon_names=["E", "L"], loss=self.parameters["insertion_loss"])
        self.do_lbb_on_photons(lbb.mode_loss, photon_names=["E", "L"], loss=self.parameters["link_loss"])
        self.do_lbb_on_photons(lbb.mode_loss, photon_names=["E", "L"], loss=self.parameters["insertion_loss"])
        self.do_lbb(
            lbb.conditional_amplitude_reflection_time_bin_spi,
            photon_early_name="E",
            photon_late_name="L",
            spin_name="Bob",
        )
        self.do_lbb_on_photons(lbb.mode_loss, photon_names=["E", "L"], loss=self.parameters["insertion_loss"])
        self.do_lbb(lbb.basis_rotation, photon_names=["E", "L"])
