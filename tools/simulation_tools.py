import numpy as np
import lib.quantum_optical_modelling as qom
import multiprocessing as multi
import functools
import time
from tqdm.notebook import tqdm

def fidelity_helper(x, kappa_in, kappa_loss, gamma, g):
    cavity_params['kappa_in']=kappa_in
    cavity_params['kappa_loss']=kappa_loss
    cavity_params['gamma']=gamma
    cavity_params['g']=g
    
    cavity_params['f_operation']=x[0]
    cavity_params['delta']=x[1]
    cavity_params['splitting']=x[2]
    F,rate = LinearReflectionProtocol(cavity_params, link_params)
    return -F

def estimate_sweep_time(sweep_dict,protocol, params):
    t0=time.time()
    protocol(params)
    tf=time.time()-t0

    par_dimensions = [len(sweep_dict[x]['range']) for x in list(sweep_dict.keys())]

    return tf*np.prod(par_dimensions)

def update_pars_and_run(protocol, params, par_names, *args):
    for i,el in enumerate(par_names):
        params[el] = args[i]
    return protocol(params)

def multiprocess_sweep(sweep_dict, protocol, params, chunksize=1):
    par_names = list(sweep_dict.keys())

    wrap = functools.partial(update_pars_and_run, protocol,params,par_names)

    pars = [sweep_dict[x]['range'] for x in par_names]
    par_dimensions = [len(x) for x in pars]
    grid = np.array(np.meshgrid(*pars))
    XY = np.array([*[x.flatten() for x in grid]]).T
    args=list(XY)

    ncpu = multi.cpu_count()
    t0=time.time()
    with multi.Pool(ncpu) as processing_pool:
        # accumulate results in a dictionary
        results = processing_pool.starmap(wrap, args, chunksize=chunksize)
    t_sim=time.time() - t0
    print('Sweep time with multi was {:.3f} s'.format(t_sim))

    
    ff = np.array([x[0] for x in results]).reshape(*par_dimensions).T
    rr = np.array([x[1] for x in results]).reshape(*par_dimensions).T

    return ff,rr

def multiprocess_async_sweep(sweep_dict, protocol, params):
    par_names = list(sweep_dict.keys())

    wrap = functools.partial(update_pars_and_run, protocol,params,par_names)

    pars = [sweep_dict[x]['range'] for x in par_names]
    grid = np.array(np.meshgrid(*pars))
    XY = np.array([*[x.flatten() for x in grid]]).T
    args=list(XY)

    ncpu = multi.cpu_count()
    t0=time.time()
    with multi.Pool(ncpu) as processing_pool:
        # accumulate results in a dictionary
        results = tqdm(processing_pool.starmap_async(wrap, args, chunksize=10), total=len(pars[0]))
    print(results.get())
    t_sim=time.time() - t0
    print('Sweep time with multi was {:.3f} s'.format(t_sim))

    par_dimensions = [len(x) for x in pars]
    ff = np.array([x[0] for x in results.get()]).reshape(*par_dimensions).T
    rr = np.array([x[1] for x in results.get()]).reshape(*par_dimensions).T

    return ff,rr


def maximize_fidelity(x0, kappa_in, kappa_loss, gamma, g, bounds = [(-55e9,-45e9), (45e9,55e9), (0.5e9, 0.8e9)]):
#     result = shgo(fidelity_helper, bounds=bounds, args = (kappa_in, kappa_loss, gamma, g), 
#        minimizer_kwargs={'method':'powell'}, options={"jac":True})
#     result = minimize(fidelity_helper, x0=x0, method='powell', args=(kappa_in, kappa_loss, gamma, g),
#                      bounds=bounds, tol=1e-5)
#     result = dual_annealing(fidelity_helper, bounds=bounds, args = (kappa_in, kappa_loss, gamma, g))
    result = brute(fidelity_helper, ranges=bounds, args = (kappa_in, kappa_loss, gamma, g), Ns=10, full_output=True)
    return result

def two_params_sweep(params_dict, protocol, protocol_params, fidelity_rate_curve=False):
    par1_name = list(params_dict.keys())[0]
    par2_name = list(params_dict.keys())[1]

    par1_range = params_dict[par1_name]['range']
    par2_range = params_dict[par2_name]['range']

    FF=[]
    RR=[]

    for p1 in par1_range:
        protocol_params[par1_name] = p1
        ff=[]
        rr=[]

        for p2 in par2_range:
            protocol_params[par2_name] = p2
            f, r = protocol(protocol_params)

            ff.append(f)
            rr.append(r)

        FF.append(ff)
        RR.append(rr)

    if fidelity_rate_curve:
        return get_fidelity_rate_curve(np.array(FF), np.array(RR))
    else:
        return np.array(FF), np.array(RR)


def three_params_sweep(params_dict, protocol, protocol_params, fidelity_rate_curve=False):
    par1_name = list(params_dict.keys())[0]
    par2_name = list(params_dict.keys())[1]
    par3_name = list(params_dict.keys())[2]

    #print('Sweeping parameters {} and {}'.format(par1_name, par2_name) )

    par1_range = params_dict[par1_name]['range']
    par2_range = params_dict[par2_name]['range']
    par3_range = params_dict[par3_name]['range']

    FF=[]
    RR=[]

    for p1 in par1_range:
        protocol_params[par1_name] = p1
        f1=[]
        r1=[]

        for p2 in par2_range:

            f2=[]
            r2=[]
            protocol_params[par2_name] = p2

            for p3 in par3_range:
                protocol_params[par3_name] = p3
                f, r = protocol(protocol_params)

                f2.append(f)
                r2.append(r)
            f1.append(f2)
            r1.append(r2)
        FF.append(f1)
        RR.append(r1)

    if fidelity_rate_curve:
        return get_fidelity_rate_curve(np.array(FF), np.array(RR))
    else:
        return np.array(FF), np.array(RR)


def multiparam_sweep(params_dict, protocol, protocol_params, fidelity_rate_curve = False):
    dim_sweep = len(params_dict)

    FF=[]
    RR=[]
    for par in params_dict:
        sweep_range = params_dict[par]['range']

        for x in sweep_range:
            protocol_params[par] = x
            ff=[]
            rr=[]
            for par2 in params_dict:
                if par2 is par:
                    pass

                sweep_range2 = params_dict[par2]['range']

                for y in sweep_range2:
                    protocol_params[par2] = y
                    fid,rate=protocol(protocol_params)
                    ff.append(fid)
                    rr.append(rate)
            FF.append(ff)
            RR.append(rr)

    if fidelity_rate_curve:
        return get_fidelity_rate_curve(np.array(FF), np.array(RR))
    else:
        return np.array(FF), np.array(RR)

def get_fidelity_rate_curve(FF, RR, n=100, type_axis='lin', rate_range=None):

    if rate_range is not None:
        rmin=rate_range[0]
        rmax=rate_range[-1]
    else:
        rmin=np.min(RR)
        rmax=np.max(RR)
    if type_axis == 'log':
        rates = np.logspace(rmin, rmax, n)
    else:
        rates = np.linspace(rmin, rmax, n)
    fids = []
    for r in rates:
        x= FF[np.where(RR>=r)]
        fids.append(np.max(x))
    return np.array(fids), rates

def get_best_fidelity_for_rate(FF, RR, rate=1e-4):
    RR = np.array(RR)
    FF = np.array(FF)
    idx = np.where(RR>=rate)
    if len(idx[0])==0:
        idx = None
    x= FF[idx]
    fid = np.max(x)
    return fid


def find_best_cavity_contrast(cavity_params, range_omega = [-70e9, 80e9], atom_centered=False):
    kappa_r=cavity_params['kappa_in']
    kappa_t=cavity_params['kappa_loss']
    kappa_loss=0
    gamma=cavity_params['gamma']
    delta=cavity_params['delta']
    splitting=cavity_params['splitting']
    g=cavity_params['g']
    #C=g**2/ ( (kappa_t+kappa_r+kappa_loss)*gamma)
    f_oper=cavity_params['f_operation']

    gamma_dephasing=cavity_params.pop('gamma_dephasing', 0)
    C=cavity_params.get('C', 4*g**2/(kappa_t+kappa_r+kappa_loss)/(gamma+gamma_dephasing))

    print(C)
    omegas=np.linspace(range_omega[0], range_omega[1], 2001)

    if atom_centered:
        range_omega = [-15e9, 1e9]
        omegas=np.linspace(range_omega[0], range_omega[1], 2001)
        t_u, r_u, l_u = qom.cavity_qom_atom_centered(omegas, -delta, kappa_r, kappa_t, kappa_loss, gamma, C, gamma_dephasing=gamma_dephasing)
        t_d, r_d, l_d = qom.cavity_qom_atom_centered(omegas+splitting/2, -delta-splitting/2, kappa_r, kappa_t, kappa_loss, gamma, C, gamma_dephasing=gamma_dephasing)
        omegas += splitting/2
    else:
        t_u, r_u, l_u = qom.cavity_qom_cavity_centered(omegas, delta, kappa_r, kappa_t, kappa_loss, gamma, C, gamma_dephasing=gamma_dephasing)
        t_d, r_d, l_d = qom.cavity_qom_cavity_centered(omegas, delta-splitting, kappa_r, kappa_t, kappa_loss, gamma, C, gamma_dephasing=gamma_dephasing)
    
    print(np.max(abs(r_u)**2-abs(r_d)**2))
    return omegas[np.argmax(abs(r_u)**2-abs(r_d)**2)]

