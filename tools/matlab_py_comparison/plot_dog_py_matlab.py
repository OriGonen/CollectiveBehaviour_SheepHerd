import matplotlib.pyplot as plt
from movement_algorithms.vivek_model import herding_model
from utils.utils import load_matlab_herding_data, transform_matlab_all_runs, extract_initial_conditions

matlab_file = "../../data/hm_1_14_new.mat"

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

pos_s_log = []
pos_d_log = []
vel_s_log = []
vel_d_log = []
spd_d_log = []

for run_idx in range(num_runs):
    pos_s_matlab = pos_s_matlab_all[run_idx]
    pos_d_matlab = pos_d_matlab_all[run_idx]
    vel_s_matlab = vel_s_matlab_all[run_idx]
    vel_d_matlab = vel_d_matlab_all[run_idx]

    initial_pos_s, initial_pos_d = extract_initial_conditions(pos_s_matlab, pos_d_matlab)
    initial_vel_s, initial_vel_d = extract_initial_conditions(vel_s_matlab, vel_d_matlab)

    print("Running Python model with extracted initial conditions...")
    pos_s_python, pos_d_python, vel_s_python, vel_d_python, spd_d_python, \
        collect_python, drive_python, force_slow_python = herding_model(
        **params,
        initial_pos_s=initial_pos_s,
        initial_pos_d=initial_pos_d,
        initial_vel_s=initial_vel_s,
        initial_vel_d=initial_vel_d
    )
    pos_s_log.append(pos_s_python)
    pos_d_log.append(pos_d_python)
    vel_s_log.append(vel_s_python)
    vel_d_log.append(vel_d_python)
    spd_d_log.append(spd_d_python)

plt.figure(figsize=(12, 10))
for idx in range(num_runs):
    pos_d = pos_d_log[idx]
    plt.plot(pos_d[:, 0], pos_d[:, 1], label=f'Run Python{idx + 1}', linewidth=2, alpha=0.8)

for run_idx in range(num_runs):
    pos_d = pos_d_matlab_all[run_idx]
    plt.plot(pos_d[:, 0], pos_d[:, 1], label=f'Run Matlab {run_idx + 1}', linewidth=2, alpha=0.8)

plt.xlabel('X Position', fontsize=12)
plt.ylabel('Y Position', fontsize=12)
plt.title(f'Dog Trajectories - All {num_runs} Runs', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.savefig('dog_traj_matlab_py.png', dpi=150)
plt.show()