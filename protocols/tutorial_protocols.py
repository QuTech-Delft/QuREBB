import qutip as qt

import lib.LBB as lbb
import lib.NQobj as nq
import lib.states as st
from lib.protocol import Protocol

def no_vacuum_projector(name, dim):
    identity = nq.name(qt.qeye(dim), name, "oper")
    vacuum = nq.name(st.vacuum_dm(dim), name, "oper")
    return identity - vacuum

class ProtocolA(Protocol):
    def __init__(self, parameters: dict):
        super().__init__(parameters)
        
        self.dm_init = nq.tensor(
            nq.name(st.alpha_dm(self.parameters['alpha']), "Alice", "state"),
            nq.name(st.alpha_dm(self.parameters['alpha']), "Bob", "state")
        )

        state_ud = nq.tensor(nq.name(  st.up, "Alice", "state"), nq.name(st.down, "Bob", "state")) 
        state_du = nq.tensor(nq.name(st.down, "Alice", "state"), nq.name(  st.up, "Bob", "state")) 
        self.target_state = (state_ud + state_du).unit()

        self.herald_projector = no_vacuum_projector("Pa", self.dim) + no_vacuum_projector("Pa_incoh", self.dim) + no_vacuum_projector("Pa_incoh2", self.dim)

    def protocol_sequence(self):
        self.do_lbb(lbb.SpontaneousEmissionFockSPI, spin_name='Alice', photon_name='Pa')
        self.do_lbb_on_photons(lbb.ModeLoss, photon_names = ["Pa", "Pa_incoh"], loss=self.parameters["insertion_loss"])

        self.do_lbb(lbb.SpontaneousEmissionFockSPI, spin_name='Bob', photon_name='Pb')
        self.do_lbb_on_photons(lbb.ModeLoss, photon_names = ["Pb", "Pb_incoh"], loss=self.parameters["insertion_loss"])

        self.do_lbb_on_photons(lbb.ModeLoss, photon_names = ["Pa", "Pa_incoh", "Pb", "Pb_incoh"], loss=self.parameters["link_loss"]/2)

        self.do_lbb(lbb.HOM, photon_names=['Pa', 'Pb'])
        self.do_lbb(lbb.HOM, photon_names=['Pa_incoh', 'Pb_incoh2'])
        self.do_lbb(lbb.HOM, photon_names=['Pb_incoh', 'Pa_incoh2'])

        self.do_lbb_on_photons(lbb.DarkCounts, photon_names = ["Pa", "Pb"])

        self.do_lbb(lbb.Herald, herald_projector=self.herald_projector)

class ProtocolB(Protocol):
    def __init__(self, parameters: dict):
        super().__init__(parameters)
        
        if "alpha" in self.parameters:
            photon_basis = qt.coherent(self.dim, self.parameters["alpha"])
        else:
            photon_basis = st.photon(self.dim)
        Ea = nq.name(photon_basis, "Ea", "state")
        La = nq.name(photon_basis, "La", "state")
        Eb = nq.name(photon_basis, "Eb", "state")
        Lb = nq.name(photon_basis, "Lb", "state")
        self.dm_init = nq.tensor(
            nq.name(st.x_dm, "Alice", "state"),
            nq.name(st.x_dm, "Bob", "state"),
            nq.ket2dm(nq.tensor((Ea + La).unit(), (Eb + Lb).unit()))
        )

        state_ud = nq.tensor(nq.name(  st.up, "Alice", "state"), nq.name(st.down, "Bob", "state")) 
        state_du = nq.tensor(nq.name(st.down, "Alice", "state"), nq.name(  st.up, "Bob", "state")) 
        self.target_state = (state_ud - state_du).unit()

        self.herald_projector = nq.tensor(no_vacuum_projector("Ea", self.dim), no_vacuum_projector("Lb", self.dim))

    def protocol_sequence(self):
        self.do_lbb_on_photons(lbb.ModeLoss, photon_names = ["Ea", "La"], loss=self.parameters["insertion_loss"])
        self.do_lbb(lbb.ConditionalAmplitudeReflectionTimeBinSPI, photon_early_name='Ea', photon_late_name='La', spin_name="Alice")
        self.do_lbb_on_photons(lbb.ModeLoss, photon_names = ["Ea", "La"], loss=self.parameters["insertion_loss"])

        self.do_lbb_on_photons(lbb.ModeLoss, photon_names = ["Eb", "Lb"], loss=self.parameters["insertion_loss"])
        self.do_lbb(lbb.ConditionalAmplitudeReflectionTimeBinSPI, photon_early_name='Eb', photon_late_name='Lb', spin_name="Bob")
        self.do_lbb_on_photons(lbb.ModeLoss, photon_names = ["Eb", "Lb"], loss=self.parameters["insertion_loss"])

        self.do_lbb_on_photons(lbb.ModeLoss, photon_names = ["Ea", "Eb", "La", "Lb"], loss=self.parameters["link_loss"]/2)

        self.do_lbb(lbb.HOM, photon_names=['Ea', 'Eb'])
        self.do_lbb(lbb.HOM, photon_names=['La', 'Lb'])

        self.do_lbb_on_photons(lbb.DarkCounts, photon_names = ["Ea", "La", "Eb", "Lb"])

        self.do_lbb(lbb.Herald, herald_projector=self.herald_projector)

class ProtocolC(Protocol):
    def __init__(self, parameters: dict):
        super().__init__(parameters)

        if "alpha" in self.parameters:
            photon_basis = qt.coherent(self.dim, self.parameters["alpha"])
        else:
            photon_basis = st.photon(self.dim)
        E = nq.name(photon_basis, "E", "state")
        L = nq.name(photon_basis, "L", "state")
        self.dm_init = nq.tensor(
            nq.name(st.x_dm, "Alice", "state"),
            nq.name(st.x_dm, "Bob", "state"),
            nq.ket2dm((E + L).unit())
        )

        self.herald_projector = no_vacuum_projector(name="E", dim=self.dim)

        state_uu = nq.tensor(nq.name(  st.up, "Alice", "state"), nq.name(  st.up, "Bob", "state")) 
        state_dd = nq.tensor(nq.name(st.down, "Alice", "state"), nq.name(st.down, "Bob", "state")) 
        self.target_state = (state_uu + state_dd).unit()

    def protocol_sequence(self):
        self.do_lbb_on_photons(lbb.ModeLoss, photon_names = ["E", "L"], loss=self.parameters["insertion_loss"])
        self.do_lbb(lbb.ConditionalAmplitudeReflectionTimeBinSPI, photon_early_name='E', photon_late_name='L', spin_name="Alice")
        self.do_lbb_on_photons(lbb.ModeLoss, photon_names = ["E", "L"], loss=self.parameters["insertion_loss"])
        
        self.do_lbb_on_photons(lbb.ModeLoss, photon_names = ["E", "L"], loss=self.parameters["link_loss"])

        self.do_lbb_on_photons(lbb.ModeLoss, photon_names = ["E", "L"], loss=self.parameters["insertion_loss"])
        self.do_lbb(lbb.ConditionalAmplitudeReflectionTimeBinSPI, photon_early_name='E', photon_late_name='L', spin_name="Bob")
        self.do_lbb_on_photons(lbb.ModeLoss, photon_names = ["E", "L"], loss=self.parameters["insertion_loss"])

        self.do_lbb(lbb.BasisRotation, photon_names=['E', 'L'])

        self.do_lbb_on_photons(lbb.DarkCounts, photon_names = ["E", "L"], )

        self.do_lbb(lbb.Herald, herald_projector=self.herald_projector)

