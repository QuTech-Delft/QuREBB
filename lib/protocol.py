import functools
import itertools
import multiprocessing as multi
import time

import numpy as np
import xarray as xr
from copy import copy
import datetime
import lib.NQobj as nq
from os.path import join

class Protocol:
    """
    This class handles a two-qubit entanglement protocol,
    holds the protocol parameters, photon encoding and density matrix

    Attributes:
            name: (str) the protocol name.
            parmeters: (dict) dictionary containing the parameters necessary for protocol operation.
            dim: (int) dimension of the photon space. Default is 3 as this is the minimum for using single photons and HOM interference.

    Additional arguments:
            photon_names: (list) list of names for the photonic modes. NB: should this be a mandatory attribute?
            start_state: (NQobj) start state of the protocol
            target_state: (NQobj) the target final state of the protocol (used to calculate fidelity)

    """

    def __init__(self, parameters: dict):

        self.parameters: dict = parameters

        self.dm: nq.NQobj = None
        self.dm_init: nq.NQobj = None
        if "dim" in parameters: # photonic size as attribute for convenience
            self.dim = parameters['dim']
        else:
            raise ValueError("The entry 'dim' needs to be present in parameters")

        self.target_state: nq.NQobj = None  # target state of spins
        self.herald_projector: nq.NQobj = None  # herald state of photon

        self.photon_names: List[str] = None  # needed for convienent handling

        self.fidelity = None
        self.rate = None

    def run(self):
        """Runs the protocol sequence and return the metrics"""
        self.dm = self.dm_init
        self.protocol_sequence()
        return self.metrics()

    def protocol_sequence():
        """Defines the protocol sequence. Should be implemented by child class."""
        pass

    def do_lbb(self, LBB, **kwargs):
        """Makes a logical building block act on the density matrix."""
        kwargs.update(self.parameters)

        self.dm = LBB(
            dm_in = self.dm,
            **kwargs
        )

    def do_lbb_on_photons(self, LBB, photon_names, **kwargs):
        for photon_name in photon_names:
            self.do_lbb(LBB, photon_name=photon_name, **kwargs)

    def metrics(self, target_state=None):
        """
        Calculates fidelity and success probability versus target spin state
        """
        if target_state is None:
            target_state = self.target_state
        fidelity = nq.fidelity(self.dm.unit(), nq.ket2dm(target_state)) ** 2
        rate = self.dm.tr()
        self.fidelity = fidelity
        self.rate = rate
        return fidelity, rate

class ProtocolSweep:
      
    def __init__(
        self, 
        protocol, 
        parameters, 
        sweep_parameters,
        save_results = False,
        save_folder = None,
        save_name = "dataset"
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
        update_parameters = {name: value for name, value in zip(sweep_parameter_names, args)}
        parameters.update(update_parameters)
        protocol = self.protocol(parameters=parameters)
        return protocol.run()

    def multiprocess_sweep(self):
        sweep_parameter_names = list(self.sweep_parameters.keys())
        wrap = functools.partial(self.update_parameters_and_run, sweep_parameter_names)

        parameter_lists = list(self.sweep_parameters.values())
        data_array_size = [len(parameter_list) for parameter_list in parameter_lists]
        parameter_values_iter = itertools.product(*[list(array) for array in parameter_lists])

        t0=time.time()
        with multi.Pool() as processing_pool:
            results = processing_pool.starmap(wrap, parameter_values_iter)
        t_sim=time.time() - t0
        print('Sweep time with multi was {:.3f} s'.format(t_sim))

        fidelity = np.array([x[0] for x in results]).reshape(data_array_size)
        rate = np.array([x[1] for x in results]).reshape(data_array_size)

        return fidelity, rate

    def run(self):
        sweep_parameter_names = list(self.sweep_parameters.keys())
        fidelity, rate = self.multiprocess_sweep()
        data_vars = {
            "fidelity": (sweep_parameter_names, fidelity), 
            "rate": (sweep_parameter_names, rate)
            }
        parameters = copy(self.parameters)
        for parameter in self.sweep_parameters:
            parameters.pop(parameter)
        self.dataset = xr.Dataset(data_vars, self.sweep_parameters, attrs=parameters)
        if self.save_results:
            self.save_dataset()

    def save_dataset(self):
        date_time = self._generate_date_time()
        file_path = join(self.save_folder, date_time + self.save_name + ".hdf5")
        self.dataset.to_netcdf(file_path, engine='h5netcdf')

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
        t0 = time.time()
        protocol.run()
        tf = time.time() - t0
        par_dimensions = [len(sweep) for sweep in self.sweep_parameters.values()]
        return tf * np.prod(par_dimensions)

    def generate_fidelity_rate_curve(self, n=100, type_axis="lin", rate_range=None):
        if self.dataset == xr.Dataset():
            raise RuntimeError("First run the sweep to create a dataset.")

        if rate_range is not None:
            rmin = rate_range[0]
            rmax = rate_range[-1]
        else:
            rmin = float(self.dataset.rate.min())
            rmax = float(self.dataset.rate.max())
        if type_axis == "log":
            rates = np.logspace(rmin, rmax, n)
        elif type_axis == 'lin':
            rates = np.linspace(rmin, rmax, n)
        else:
            raise ValueError('type_axis should be lin or log')
        
        fidelities = []
        for rate in rates:
            rate_above_bound = (self.dataset.fidelity == self.dataset.fidelity.where(self.dataset.rate >= rate).max())
            fidelity = self.dataset.fidelity.groupby(rate_above_bound)[True][0] # select only single value
            for coord in fidelity.coords:
                if "stacked" in coord:
                    fidelity = fidelity.reset_coords(coord, drop=True)
            fidelities.append(fidelity.expand_dims("rate").assign_coords(rate=[rate]))
        self.dataset_fidelity_rate = xr.combine_by_coords(fidelities)
        