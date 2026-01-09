from pathlib import Path
import numpy as np
from movement_algorithms.fatigue_model import herding_model
from utils.utils import save_simulation_results

def run_simulations_ftm(n_runs, no_shp, box_length, rad_rep_s, rad_rep_dog, K_atr, k_atr,
                        k_alg, vs, v_dog, h, rho_a, rho_d, e, c, alg_str, f_n,
                        pd, pc, n_iter, delta_t, F_i, R_i, TL_max, TL_chase,
                        L_D, L_R, epsilon_v, TL_gather, TL_drive, TL_idle,
                        seed=None):
    """
    Run multiple simulations of the fatigue herding model and collect results.
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize storage arrays for all runs
    all_pos_s = np.zeros((n_runs, n_iter, no_shp, 2))
    all_vel_s = np.zeros((n_runs, n_iter, no_shp, 2))
    all_pos_d = np.zeros((n_runs, n_iter, 2))
    all_vel_d = np.zeros((n_runs, n_iter, 2))
    all_spd_s = np.zeros((n_runs, n_iter, no_shp))
    all_spd_d = np.zeros((n_runs, n_iter))
    all_collect_t = np.zeros((n_runs, n_iter))
    all_drive_t = np.zeros((n_runs, n_iter))
    all_force_slow_t = np.zeros((n_runs, n_iter))

    # Fatigue storage
    all_M_A_s = np.zeros((n_runs, n_iter, no_shp))
    all_M_F_s = np.zeros((n_runs, n_iter, no_shp))
    all_M_R_s = np.zeros((n_runs, n_iter, no_shp))
    all_TL_s = np.zeros((n_runs, n_iter, no_shp))
    all_C_s = np.zeros((n_runs, n_iter, no_shp))

    all_M_A_d = np.zeros((n_runs, n_iter))
    all_M_F_d = np.zeros((n_runs, n_iter))
    all_M_R_d = np.zeros((n_runs, n_iter))
    all_TL_d = np.zeros((n_runs, n_iter))
    all_C_d = np.zeros((n_runs, n_iter))

    print(f"Running {n_runs} FTM simulations with {n_iter} iterations each...")

    for run in range(n_runs):
        if (run + 1) % max(1, n_runs // 10) == 0:
            print(f"  Completed {run + 1}/{n_runs} runs")

        res = herding_model(
            no_shp=no_shp, box_length=box_length, rad_rep_s=rad_rep_s,
            rad_rep_dog=rad_rep_dog, K_atr=K_atr, k_atr=k_atr,
            k_alg=k_alg, vs=vs, v_dog=v_dog, h=h, rho_a=rho_a, rho_d=rho_d,
            e=e, c=c, alg_str=alg_str, f_n=f_n, pd=pd, pc=pc,
            n_iter=n_iter, delta_t=delta_t, F_i=F_i, R_i=R_i,
            TL_max=TL_max, TL_chase=TL_chase, L_D=L_D, L_R=L_R,
            epsilon_v=epsilon_v, TL_gather=TL_gather, TL_drive=TL_drive,
            TL_idle=TL_idle
        )

        all_pos_s[run] = res["pos_s_dat"]
        all_pos_d[run] = res["pos_d_dat"]
        all_vel_s[run] = res["vel_s_dat"]
        all_vel_d[run] = res["vel_d_dat"]
        all_spd_s[run] = res["spd_s_dat"]
        all_spd_d[run] = res["spd_d_dat"]
        all_collect_t[run] = res["collect_t"]
        all_drive_t[run] = res["drive_t"]
        all_force_slow_t[run] = res["force_slow_t"]

        all_M_A_s[run] = res["M_A_s_dat"]
        all_M_F_s[run] = res["M_F_s_dat"]
        all_M_R_s[run] = res["M_R_s_dat"]
        all_TL_s[run] = res["TL_s_dat"]
        all_C_s[run] = res["C_s_dat"]

        all_M_A_d[run] = res["M_A_d_dat"]
        all_M_F_d[run] = res["M_F_d_dat"]
        all_M_R_d[run] = res["M_R_d_dat"]
        all_TL_d[run] = res["TL_d_dat"]
        all_C_d[run] = res["C_d_dat"]

    print("All simulations completed!")

    results = {
        'pos_s': all_pos_s,
        'vel_s': all_vel_s,
        'pos_d': all_pos_d,
        'vel_d': all_vel_d,
        'spd_s': all_spd_s,
        'spd_d': all_spd_d,
        'collect_t': all_collect_t,
        'drive_t': all_drive_t,
        'force_slow_t': all_force_slow_t,
        'M_A_s': all_M_A_s,
        'M_F_s': all_M_F_s,
        'M_R_s': all_M_R_s,
        'TL_s': all_TL_s,
        'C_s': all_C_s,
        'M_A_d': all_M_A_d,
        'M_F_d': all_M_F_d,
        'M_R_d': all_M_R_d,
        'TL_d': all_TL_d,
        'C_d': all_C_d,
        'no_runs': n_runs,
        'model': 'FTM',
        'params': {
            'no_shp': no_shp, 'box_length': box_length, 'rad_rep_s': rad_rep_s,
            'rad_rep_dog': rad_rep_dog, 'K_atr': K_atr, 'k_atr': k_atr,
            'k_alg': k_alg, 'vs': vs, 'v_dog': v_dog, 'h': h, 'rho_a': rho_a,
            'rho_d': rho_d, 'e': e, 'c': c, 'alg_str': alg_str, 'f_n': f_n,
            'pd': pd, 'pc': pc, 'n_iter': n_iter, 'delta_t': delta_t,
            'F_i': F_i, 'R_i': R_i, 'TL_max': TL_max, 'TL_chase': TL_chase,
            'L_D': L_D, 'L_R': L_R, 'epsilon_v': epsilon_v,
            'TL_gather': TL_gather, 'TL_drive': TL_drive, 'TL_idle': TL_idle,
            'seed': seed
        }
    }
    return results

if __name__ == "__main__":
    n_runs = 100 # Reduced for faster testing, adjust as needed
    n_iter = 370
    num_sheep = 14

    filename = f'ftm_{num_sheep}_{n_runs}_{n_iter}'
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
        n_iter=n_iter,
        delta_t=1,
        F_i=0.1,
        R_i=0.02,
        TL_max=0.8,
        TL_chase=0.9,
        L_D=10,
        L_R=10,
        epsilon_v=0.1,
        TL_gather=0.6,
        TL_drive=0.6,
        TL_idle=0.05
    )

    if not Path("../data").exists():
        print(f"Data directory was not found")
        exit(1)

    results = run_simulations_ftm(**params, n_runs=n_runs)
    save_simulation_results(results, '../data/' + filename)
