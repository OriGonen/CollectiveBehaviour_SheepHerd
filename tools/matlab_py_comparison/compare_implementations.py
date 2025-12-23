import numpy as np
from pathlib import Path
from utils.utils import load_matlab_herding_data, transform_matlab_all_runs, extract_initial_conditions
from movement_algorithms.vivek_model import herding_model


def compute_trajectory_distance(pos_matlab, pos_python):
    return np.sqrt(np.sum((pos_matlab - pos_python) ** 2, axis=-1))


def compute_statistics(matlab_data, python_data):
    flat_matlab = matlab_data.flatten()
    flat_python = python_data.flatten()
    correlation = np.corrcoef(flat_matlab, flat_python)[0, 1]

    return {
        'correlation': correlation,
    }

def analyze_single_run(pos_s_matlab, pos_d_matlab, pos_s_python, pos_d_python, run_idx):
    print(f"\n{'=' * 60}")
    print(f"Run {run_idx + 1} Analysis")
    print(f"{'=' * 60}")

    sheep_stats = compute_statistics(pos_s_matlab, pos_s_python)
    dog_stats = compute_statistics(pos_d_matlab, pos_d_python)

    print("\nSheep Position Statistics:")
    print(f"  Correlation: {sheep_stats['correlation']:.6f}")

    print("\nDog Position Statistics:")
    print(f"  Correlation: {dog_stats['correlation']:.6f}")

    return sheep_stats, dog_stats


def main():
    matlab_file = "../../data/hm_1_14_det.mat"

    if not Path(matlab_file).exists():
        print(f"Error: {matlab_file} not found")
        return

    print("Loading Matlab data...")
    data = load_matlab_herding_data(matlab_file)
    params = data['params']

    pos_s_matlab_all, pos_d_matlab_all, vel_s_matlab_all, vel_d_matlab_all, spd_d_matlab_all = \
        transform_matlab_all_runs(
            data['pos_s'],
            data['pos_d'],
            data['vel_s'],
            data['vel_d'],
            data['spd_d']
        )

    num_runs = pos_s_matlab_all.shape[0]
    print(f"Detected {num_runs} simulation run(s)")
    print(f"Number of timesteps: {params['n_iter']}")
    print(f"Number of sheep: {params['no_shp']}")

    all_sheep_stats = []
    all_dog_stats = []

    for run_idx in range(num_runs):
        print(f"\n{'#' * 60}")
        print(f"Processing Run {run_idx + 1}/{num_runs}")
        print(f"{'#' * 60}")

        pos_s_matlab = pos_s_matlab_all[run_idx]
        pos_d_matlab = pos_d_matlab_all[run_idx]
        vel_s_matlab = vel_s_matlab_all[run_idx]
        vel_d_matlab = vel_d_matlab_all[run_idx]

        initial_pos_s, initial_pos_d = extract_initial_conditions(pos_s_matlab, pos_d_matlab)
        initial_vel_s, initial_vel_d = extract_initial_conditions(vel_s_matlab, vel_d_matlab)

        print("Running Python model with extracted initial conditions...")
        pos_s_python, pos_d_python, vel_s_python, vel_d_python, spd_d_python, \
            collect_python, drive_python, force_slow_python = herding_model(
            no_shp=params['no_shp'],
            box_length=params['box_length'],
            rad_rep_s=params['red_rep_s'],
            rad_rep_dog=params['rad_rep_dog'],
            K_atr=params['K_atr'],
            k_atr=params['k_atr'],
            k_alg=params['k_alg'],
            vs=params['vs'],
            v_dog=params['v_dog'],
            h=params['h'],
            rho_a=params['rho_a'],
            rho_d=params['rho_d'],
            e=params['e'],
            c=params['c'],
            alg_str=params['alg_str'],
            f_n=params['f_n'],
            pd=params['pd'],
            pc=params['pc'],
            n_iter=params['n_iter'],
            initial_pos_s=initial_pos_s,
            initial_pos_d=initial_pos_d,
            initial_vel_s=initial_vel_s,
            initial_vel_d=initial_vel_d
        )

        sheep_stats, dog_stats = analyze_single_run(
            pos_s_matlab, pos_d_matlab,
            pos_s_python, pos_d_python,
            run_idx
        )

        all_sheep_stats.append(sheep_stats)
        all_dog_stats.append(dog_stats)

    if num_runs > 1:
        print(f"\n{'=' * 60}")
        print("Aggregate Statistics Across All Runs")
        print(f"{'=' * 60}")

        sheep_corr_mean = np.mean([s['correlation'] for s in all_sheep_stats])
        dog_corr_mean = np.mean([s['correlation'] for s in all_dog_stats])

        print("\nSheep (across all runs):")
        print(f"  Correlation: {sheep_corr_mean:.6f}")

        print("\nDog (across all runs):")
        print(f"  Correlation: {dog_corr_mean:.6f}")

if __name__ == "__main__":
    main()