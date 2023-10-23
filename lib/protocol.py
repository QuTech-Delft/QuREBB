import datetime
import functools
import itertools
import multiprocessing as multi
import time
from copy import copy
from os.path import join
from typing import List, Optional

import numpy as np
import qutip as qt
import xarray as xr

import lib.LBB as lbb
import lib.NQobj as nq

qt.settings.auto_tidyup = False


class Protocol:
    """
    This class handles a two-qubit entanglement protocol.
    This holds protocol parameters, photon encoding and density matrix.

    Attributes:
            name : str
                Name of a protocol.
            parmeters : dict
                Dictionary of parameters necessary for the operation of the protocol.
            dim : int
                Dimension of photonic modes.
                Default is 3 (minimum for using single photons and HOM interference).

    Additional arguments:
            photon_names : list
                List of names for the photonic modes.
            start_state : NQobj
                Initial state of the protocol
            target_state : NQobj
                Target state of the protocol (for fidelity calculation).

    """

    def __init__(self, parameters: dict):
        """
        Initialize the Protocol class.

        Parameters:
        ----------
        parameters : dict
            Dictionary of parameters of protocol.
        """

        self.parameters: dict = parameters
        self.dm: nq.NQobj = None
        self.dm_init: nq.NQobj = None
        self.dm_heralded: List[nq.NQobj] = None

        # Check for dimension of photonic mode in the parameters, else raise an error
        if "dim" in parameters:
            self.dim = parameters["dim"]
        else:
            raise ValueError("The entry 'dim' needs to be present in parameters")

        self.target_states: List[nq.NQobj] = []  # target spin state
        self.herald_projectors: List[nq.NQobj] = []  # heralding operator

        # Fidelity to be calculated
        self.fidelity: Optional[list] = None
        self.fidelity_total: Optional[float] = None

        # Entanglement generation rate to be calculated
        self.rate: Optional[list] = None
        self.rate_total: Optional[float] = None

    def run(self):
        """
        Execute the protocol sequence.

        Returns:
        -------
        tuple
            Tuple containing fidelity and rate of the protocol.
        """
        self.dm = self.dm_init
        self.protocol_sequence()
        fidelity, rate = self.herald()
        return fidelity, rate

    def protocol_sequence(self):
        """
        Defines the protocol sequence. To be implemented in subclasses.
        """
        pass

    def do_lbb(self, LBB, **kwargs):
        """
        Apply a logical building block to the density matrix.

        Parameters:
        ----------
        LBB : function of LBB.py
            Logical building block.
        **kwargs : dict
            Additional keyword arguments.
        """
        kwargs.update(self.parameters)

        self.dm = LBB(dm_in=self.dm, **kwargs)

    def do_lbb_on_photons(self, LBB, photon_names, **kwargs):
        """
        Apply a logical building block acting on photonic modes.

        Parameters:
        ----------
        LBB : function (of LBB.py)
            Logical building block.
        photon_names : list of str
            Name of photonic modes which LBB is applied to.
        **kwargs : dict
            Additional keyword arguments.
        """

        for photon_name in photon_names:
            self.do_lbb(LBB, photon_name=photon_name, **kwargs)

    def herald(self):
        """
        Perform a heralding operation --- projection --- to the density matrix.
        This method also calculates metrics for each processed matrix, and updates the
        instance's fidelity and rate attributes.

        Returns:
        -------
        tuple
            A tuple containing the fidelity and rate after heralding.
        """

        # Create a copy of the current density matrix
        dm = copy(self.dm)

        # Initialize lists to store fidelity, rate, and heralded density matrices
        fidelity = []
        rate = []
        dm_heralded = []

        # Iterate over herald projectors and target states
        for herald_projector, target_state in zip(self.herald_projectors, self.target_states):

            # Apply the herald operation to the density matrix using the given projector
            self.do_lbb(lbb.herald, herald_projector=herald_projector)

            # Calculate the fidelity and rate metrics for the current state
            metrics = self.metrics(target_state)
            fidelity.append(metrics[0])
            rate.append(metrics[1])

            # Store the heralded density matrix
            dm_heralded.append(self.dm)

            # Reset the density matrix to its original state before the next iteration
            self.dm = copy(dm)

        # Update class attributes with calculated values
        self.fidelity = fidelity
        self.fidelity_total = np.average(np.array(fidelity), weights=np.array(rate))
        self.rate = rate
        self.rate_total = sum(rate)
        self.dm_heralded = dm_heralded
        return self.fidelity_total, self.rate_total

    def metrics(self, target_state):
        """
        Calculate the fidelity and success probability of the current density matrix for a given target spin state.

        Parameters:
        ----------
        target_state : NQobj
            The target quantum state to which the fidelity of the current state is compared.

        Returns:
        -------
        tuple
            A tuple containing the fidelity and success rate of the current density matrix
            with respect to the target state.
        """

        # Calculate the fidelity between the current state and the target state
        fidelity = nq.fidelity(self.dm.unit(), nq.ket2dm(target_state)) ** 2

        # Calculate the trace (success probability) of the current density matrix
        rate = self.dm.tr()

        return fidelity, rate


class ProtocolSweep:
    def __init__(
        self, protocol, parameters, sweep_parameters, save_results=False, save_folder=None, save_name="dataset"
    ):

        self.protocol = protocol
        self.parameters = parameters
        self.sweep_parameters = sweep_parameters
        self.save_results = save_results
        self.save_folder = save_folder
        self.save_name = save_name
        if save_results:
            if save_folder is None or save_name is None:
                raise ValueError("If save_result is True, save_folder and save_name can't be None.")
        self.dataset = xr.Dataset()
        self.dataset_fidelity_rate = xr.Dataset()

    def update_parameters_and_run(self, sweep_parameter_names, *args):
        parameters = copy(self.parameters)
        update_parameters = dict(zip(sweep_parameter_names, args))
        parameters.update(update_parameters)
        protocol = self.protocol(parameters=parameters)
        return protocol.run()

    def multiprocess_sweep(self):
        sweep_parameter_names = list(self.sweep_parameters.keys())
        wrap = functools.partial(self.update_parameters_and_run, sweep_parameter_names)

        parameter_lists = list(self.sweep_parameters.values())
        data_array_size = [len(parameter_list) for parameter_list in parameter_lists]
        parameter_values_iter = itertools.product(*[list(array) for array in parameter_lists])

        time_start = time.time()
        with multi.Pool() as processing_pool:
            results = processing_pool.starmap(wrap, parameter_values_iter)
        time_sim = time.time() - time_start
        print(f"Sweep time with multi was {time_sim:.3f} s")

        fidelity = np.array([x[0] for x in results]).reshape(data_array_size)
        rate = np.array([x[1] for x in results]).reshape(data_array_size)

        return fidelity, rate

    def run(self):
        sweep_parameter_names = list(self.sweep_parameters.keys())
        fidelity, rate = self.multiprocess_sweep()
        data_vars = {"fidelity": (sweep_parameter_names, fidelity), "rate": (sweep_parameter_names, rate)}
        parameters = copy(self.parameters)
        for parameter in self.sweep_parameters:
            parameters.pop(parameter)
        self.dataset = xr.Dataset(data_vars, self.sweep_parameters, attrs=parameters)
        if self.save_results:
            self.save_dataset()

    def save_dataset(self):
        date_time = self._generate_date_time()
        file_path = join(self.save_folder, date_time + self.save_name + ".hdf5")
        # Invalid_netcdf is used to be able to save None and bools as attrs
        self.dataset.to_netcdf(file_path, engine="h5netcdf", invalid_netcdf=True)

    def save_dataset_fidelity_rate(self):
        date_time = self._generate_date_time()
        file_path = join(self.save_folder, f"{date_time}{self.save_name}_fidelity_rate.hdf5")
        # Invalid_netcdf is used to be able to save None and bools as attrs
        self.dataset_fidelity_rate.to_netcdf(file_path, engine="h5netcdf", invalid_netcdf=True)

    def _generate_date_time(self):
        time_stamp = datetime.datetime.now()
        # time_stamp gives microseconds by default
        (date_time, micro) = time_stamp.strftime("%Y%m%d-%H%M%S-.%f").split(".")
        # this ensures the string is formatted correctly as some systems return 0 for micro
        date_time = f"{date_time}{int(int(micro) / 1000):03d}-"
        return date_time

    def estimate_sweep_time(self):
        parameters = copy(self.parameters)
        for key, values in self.sweep_parameters.items():
            parameters[key] = values[0]
        protocol = self.protocol(parameters=parameters)
        time_start = time.time()
        protocol.run()
        time_single = time.time() - time_start
        par_dimensions = [len(sweep) for sweep in self.sweep_parameters.values()]
        return time_single * np.prod(par_dimensions)

    def generate_fidelity_rate_curve(self, number_of_rate_points=100, type_axis="lin", rate_range=None):
        if self.dataset == xr.Dataset():
            raise RuntimeError("First run the sweep to create a dataset.")

        if rate_range is not None:
            rmin = rate_range[0]
            rmax = rate_range[-1]
        else:
            rmin = float(self.dataset.rate.min())
            rmax = float(self.dataset.rate.max())
        if type_axis == "log":
            rates = np.geomspace(rmin, rmax, number_of_rate_points)
        elif type_axis == "lin":
            rates = np.linspace(rmin, rmax, number_of_rate_points)
        else:
            raise ValueError("type_axis should be lin or log")

        fidelities = []
        for rate in rates:
            rate_above_bound = self.dataset.fidelity == self.dataset.fidelity.where(self.dataset.rate >= rate).max()
            fidelity = self.dataset.fidelity.groupby(rate_above_bound)[True][0]  # select only single value
            for coord in fidelity.coords:
                if "stacked" in coord:
                    fidelity = fidelity.reset_coords(coord, drop=True)
            fidelities.append(fidelity.expand_dims("rate").assign_coords(rate=[rate]))
        self.dataset_fidelity_rate = xr.combine_by_coords(fidelities).assign_attrs(self.dataset.attrs)


def load_dataset(path):
    return xr.load_dataset(path, engine="h5netcdf")
