import numpy as np
from scipy.io import loadmat

from Movement_Algorithms.jadhav_model import herding_model


def load_matlab_herding_data(filename):
    data = loadmat(filename)

    pos_s = data['pos_s']
    pos_d = data['pos_d']
    vel_s = data['vel_s']
    vel_d = data['vel_d']
    spd_d = data['spd_d']

    return {
        'pos_s': pos_s,
        'pos_d': pos_d,
        'vel_s': vel_s,
        'vel_d': vel_d,
        'spd_d': spd_d,
        'no_runs': int(data['no_it'][0, 0]),
        'params': {
            'no_shp': int(data['no_shp'][0, 0]),
            'n_iter': int(data['n_iter'][0, 0]),
            'box_length': float(data['box_length'][0, 0]),
            'rad_rep_s': float(data['red_rep_s'][0, 0]),
            'rad_rep_dog': float(data['rad_rep_dog'][0, 0]),
            'K_atr': int(data['K_atr'][0, 0]),
            'k_atr': int(data['k_atr'][0, 0]),
            'k_alg': int(data['k_alg'][0, 0]),
            'vs': float(data['vs'][0, 0]),
            'v_dog': float(data['v_dog'][0, 0]),
            'h': float(data['h'][0, 0]),
            'rho_a': float(data['rho_a'][0, 0]),
            'rho_d': float(data['rho_d'][0, 0]),
            'e': float(data['e'][0, 0]),
            'c': float(data['c'][0, 0]),
            'alg_str': float(data['alg_str'][0, 0]),
            'f_n': float(data['f_n'][0, 0]),
            'pd': float(data['pd'][0, 0]),
            'pc': float(data['pc'][0, 0]),
        }
    }


def arr1d_to_scalar(arr):
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def transform_matlab_single_run(pos_s, pos_d, vel_s, vel_d, spd_d, run_idx=0):
    if pos_s.ndim == 4:
        pos_s = pos_s[:, :, :, run_idx]
        vel_s = vel_s[:, :, :, run_idx]
    if pos_d.ndim == 3:
        pos_d = pos_d[:, :, run_idx]
        vel_d = vel_d[:, :, run_idx]
        spd_d = spd_d[:, run_idx]
    if spd_d.ndim == 2:
        spd_d = spd_d[:, run_idx]

    pos_s_transformed = np.transpose(pos_s, (2, 0, 1))
    vel_s_transformed = np.transpose(vel_s, (2, 0, 1))
    pos_d_transformed = pos_d
    vel_d_transformed = vel_d
    spd_d_transformed = arr1d_to_scalar(spd_d)

    return pos_s_transformed, pos_d_transformed, vel_s_transformed, vel_d_transformed, spd_d_transformed


def load_simulation_results_matlab(filename):
    data = load_matlab_herding_data(filename)

    pos_s = data['pos_s']
    vel_s = data['vel_s']
    pos_d = data['pos_d']
    vel_d = data['vel_d']
    spd_d = data['spd_d']

    if data['pos_s'].ndim == 3:
        pos_s = pos_s[:, :, :, np.newaxis]
        vel_s = vel_s[:, :, :, np.newaxis]
        pos_d = pos_d[:, :, np.newaxis]
        vel_d = vel_d[:, :, np.newaxis]
        spd_d = spd_d[:, np.newaxis]

    pos_s_all = np.transpose(pos_s, (3, 2, 0, 1))
    vel_s_all = np.transpose(vel_s, (3, 2, 0, 1))
    pos_d_all = np.transpose(pos_d, (2, 0, 1))
    vel_d_all = np.transpose(vel_d, (2, 0, 1))
    spd_d_all = []

    for r in range(spd_d.shape[1]):
        spd_d_all.append(arr1d_to_scalar(spd_d[:, r]))

    data['vel_s'] = vel_s_all
    data['pos_s'] = pos_s_all
    data['vel_d'] = vel_d_all
    data['spd_d'] = spd_d_all
    data['pos_d'] = pos_d_all

    return data


def extract_initial_conditions(pos_s, pos_d):
    return pos_s[0], pos_d[0]


def run_simulations(n_runs, no_shp, box_length, rad_rep_s, rad_rep_dog, K_atr, k_atr,
                    k_alg, vs, v_dog, h, rho_a, rho_d, e, c, alg_str, f_n,
                    pd, pc, n_iter, seed=None):
    """
    Run multiple simulations of the herding model and collect results.

    Parameters:
    -----------
    n_runs : int
        Number of simulation runs to perform
    no_shp : int
        Number of sheep
    box_length : float
        Size of the simulation box
    rad_rep_s : float
        Repulsion radius between sheep
    rad_rep_dog : float
        Repulsion radius between dog and sheep
    K_atr : int
        Number of nearest neighbors to consider for attraction
    k_atr : int
        Number of neighbors randomly selected from K_atr for attraction
    k_alg : int
        Number of neighbors for alignment
    vs : float
        Speed of sheep
    v_dog : float
        Speed of dog
    h : float
        Inertia parameter
    rho_a : float
        Repulsion strength between sheep
    rho_d : float
        Repulsion strength from dog
    e : float
        Random noise strength
    c : float
        Attraction strength
    alg_str : float
        Alignment strength
    f_n : float
        Threshold distance for collecting behavior
    pd : float
        Distance behind center of mass for driving
    pc : float
        Distance behind furthest sheep for collecting
    n_iter : int
        Number of iterations per simulation
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    results : dict
        Dictionary containing all simulation results and parameters
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize storage arrays for all runs
    # Shape: (n_runs, n_iter, no_shp, 2) for sheep positions/velocities
    all_pos_s = np.zeros((n_runs, n_iter, no_shp, 2))
    all_vel_s = np.zeros((n_runs, n_iter, no_shp, 2))

    # Shape: (n_runs, n_iter, 2) for dog positions/velocities
    all_pos_d = np.zeros((n_runs, n_iter, 2))
    all_vel_d = np.zeros((n_runs, n_iter, 2))

    # Shape: (n_runs, n_iter) for scalar data
    all_spd_d = np.zeros((n_runs, n_iter))
    all_collect_t = np.zeros((n_runs, n_iter))
    all_drive_t = np.zeros((n_runs, n_iter))
    all_force_slow_t = np.zeros((n_runs, n_iter))

    print(f"Running {n_runs} simulations with {n_iter} iterations each...")

    for run in range(n_runs):
        if (run + 1) % max(1, n_runs // 10) == 0:
            print(f"  Completed {run + 1}/{n_runs} runs")

        # Run a single simulation
        pos_s, pos_d, vel_s, vel_d, spd_d, collect_t, drive_t, force_slow_t = herding_model(
            no_shp=no_shp,
            box_length=box_length,
            rad_rep_s=rad_rep_s,
            rad_rep_dog=rad_rep_dog,
            K_atr=K_atr,
            k_atr=k_atr,
            k_alg=k_alg,
            vs=vs,
            v_dog=v_dog,
            h=h,
            rho_a=rho_a,
            rho_d=rho_d,
            e=e,
            c=c,
            alg_str=alg_str,
            f_n=f_n,
            pd=pd,
            pc=pc,
            n_iter=n_iter
        )

        # Store results
        all_pos_s[run] = pos_s
        all_vel_s[run] = vel_s
        all_pos_d[run] = pos_d
        all_vel_d[run] = vel_d
        all_spd_d[run] = spd_d
        all_collect_t[run] = collect_t
        all_drive_t[run] = drive_t
        all_force_slow_t[run] = force_slow_t

    print("All simulations completed!")

    # Package results with parameters
    results = {
        'pos_s': all_pos_s,
        'vel_s': all_vel_s,
        'pos_d': all_pos_d,
        'vel_d': all_vel_d,
        'spd_d': all_spd_d,
        'collect_t': all_collect_t,
        'drive_t': all_drive_t,
        'force_slow_t': all_force_slow_t,
        'no_runs': n_runs,
        'params': {
            'no_shp': no_shp,
            'box_length': box_length,
            'rad_rep_s': rad_rep_s,
            'rad_rep_dog': rad_rep_dog,
            'K_atr': K_atr,
            'k_atr': k_atr,
            'k_alg': k_alg,
            'vs': vs,
            'v_dog': v_dog,
            'h': h,
            'rho_a': rho_a,
            'rho_d': rho_d,
            'e': e,
            'c': c,
            'alg_str': alg_str,
            'f_n': f_n,
            'pd': pd,
            'pc': pc,
            'n_iter': n_iter,
            'seed': seed
        }
    }

    return results


def save_simulation_results(results, filename):
    if not filename.endswith('.npz'):
        filename += '.npz'

    np.savez_compressed(filename, results=results)
    print(f"Results saved to {filename}")


def load_simulation_results(filename):
    if not filename.endswith('.npz'):
        filename += '.npz'

    data = np.load(filename, allow_pickle=True)
    results = data['results'].item()

    print(f"Loaded results from {filename}")
    print(f"  Runs: {results['no_runs']}")
    print(f"  Iterations: {results['params']['n_iter']}")
    print(f"  Sheep: {results['params']['no_shp']}")

    return results
