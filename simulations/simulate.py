from pathlib import Path

import numpy as np

from utils.utils import run_simulations, save_simulation_results

if __name__ == "__main__":
    n_runs = 300
    n_iter = 370
    num_sheep = 14

    filename = f'jadhav_{num_sheep}_{n_runs}_{n_iter}'
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

    if not Path("./data").exists():
        print(f"Data directory was not found")
        exit(1)

    results = run_simulations(**params, n_runs=n_runs)

    save_simulation_results(results, './data/' + filename)
