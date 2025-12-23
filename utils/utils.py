import numpy as np
from scipy.io import loadmat


def load_matlab_herding_data(matlab_file):
    data = loadmat(matlab_file)

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
        'params': {
            'no_shp': int(data['no_shp'][0, 0]),
            'n_iter': int(data['n_iter'][0, 0]),
            'no_it': int(data['no_it'][0, 0]),
            'box_length': float(data['box_length'][0, 0]),
            'red_rep_s': float(data['red_rep_s'][0, 0]),
            'rad_rep_dog': float(data['rad_rep_dog'][0, 0]),
            'K_atr': int(data['k_atr'][0, 0]),
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


def transform_matlab_all_runs(pos_s, pos_d, vel_s, vel_d, spd_d):
    if pos_s.ndim == 3:
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

    return pos_s_all, pos_d_all, vel_s_all, vel_d_all, spd_d_all


def extract_initial_conditions(pos_s, pos_d):
    return pos_s[0], pos_d[0]