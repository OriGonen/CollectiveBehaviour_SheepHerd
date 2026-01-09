from pathlib import Path
import numpy as np
from Movement_Algorithms.jadhav_model import herding_model
from utils.utils import save_simulation_results

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
        res = herding_model(
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
        all_pos_s[run] = res[0]
        all_pos_d[run] = res[1]
        all_vel_s[run] = res[2]
        all_vel_d[run] = res[3]
        all_spd_d[run] = res[4]
        all_collect_t[run] = res[5]
        all_drive_t[run] = res[6]
        all_force_slow_t[run] = res[7]

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

if __name__ == "__main__":
    n_runs = 300
    n_iter = 370
    num_sheep = 14

    filename = f'test_jadhav_{num_sheep}_{n_runs}_{n_iter}'
    rad_rep_s = 2
    params = dict(
        no_shp=num_sheep,
        box_length=250,
        rad_rep_s=rad_rep_s,
        rad_rep_dog=12,
        K_atr=10,
        k_atr=4,
        k_alg=1,
        vs=1,
        v_dog=1.5,
        h=0.5,
        rho_a=2,
        rho_d=1,
        e=0.3,
        c=1.5,
        alg_str=1.3,
        f_n=rad_rep_s * (num_sheep ** (2 / 3)),
        pd=rad_rep_s * np.sqrt(num_sheep),
        pc=rad_rep_s,
        n_iter=n_iter
    )

    if not Path("../data").exists():
        print(f"Data directory was not found")
        exit(1)

    results = run_simulations(**params, n_runs=n_runs)

    save_simulation_results(results, '../data/' + filename)
