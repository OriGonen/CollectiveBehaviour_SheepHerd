import numpy as np
from pathlib import Path
from utils.utils import load_matlab_herding_data, transform_matlab_all_runs, extract_initial_conditions
from movement_algorithms.vivek_model import herding_model
import scipy

def compute_pearsoncorr(matlab_data, python_data):
    pearson = scipy.stats.pearsonr(matlab_data, python_data, axis=0)

    return {
        'correlationx': pearson.correlation[0],
        'correlationy': pearson.correlation[1]
    }

def main():
    matlab_file = "../../data/hm_300_same_14.mat"

    if not Path(matlab_file).exists():
        print(f"Error: {matlab_file} not found")
        return

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

    all_dog_correlations_p_x = []
    all_dog_correlations_p_y = []

    all_dog_correlations_v_x = []
    all_dog_correlations_v_y = []

    for run_idx in range(num_runs):
        print(f"run id {run_idx + 1}")
        pos_s_matlab = pos_s_matlab_all[run_idx]
        pos_d_matlab = pos_d_matlab_all[run_idx]
        vel_s_matlab = vel_s_matlab_all[run_idx]
        vel_d_matlab = vel_d_matlab_all[run_idx]

        initial_pos_s, initial_pos_d = extract_initial_conditions(pos_s_matlab, pos_d_matlab)
        initial_vel_s, initial_vel_d = extract_initial_conditions(vel_s_matlab, vel_d_matlab)

        pos_s_python, pos_d_python, vel_s_python, vel_d_python, spd_d_python, \
            collect_python, drive_python, force_slow_python = herding_model(
            **params,
            initial_pos_s=initial_pos_s,
            initial_pos_d=initial_pos_d,
            initial_vel_s=initial_vel_s,
            initial_vel_d=initial_vel_d
        )

        dog_stats = compute_pearsoncorr(pos_d_matlab, pos_d_python)

        all_dog_correlations_p_x.append(dog_stats['correlationx'])
        all_dog_correlations_p_y.append(dog_stats['correlationy'])

        dog_stats = compute_pearsoncorr(vel_d_matlab, vel_d_python)

        all_dog_correlations_v_x.append(dog_stats['correlationx'])
        all_dog_correlations_v_y.append(dog_stats['correlationy'])

    print(f"\n{'=' * 60}")
    print("Aggregate Statistics Across All Runs")
    print(f"{'=' * 60}")

    print("\nDog (mean across all runs):")
    print(f"  P:Correlation_x: {np.mean(all_dog_correlations_p_x):.6f}")
    print(f"  P:Correlation_y: {np.mean(all_dog_correlations_p_y):.6f}")

    print(f"  V:Correlation_x: {np.mean(all_dog_correlations_v_x):.6f}")
    print(f"  V:Correlation_y: {np.mean(all_dog_correlations_v_y):.6f}")

if __name__ == "__main__":
    main()