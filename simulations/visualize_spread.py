import numpy as np
from pathlib import Path

from Movement_Algorithms.jadhav_model import herding_model
from animation import HerdingAnimation


def generate_spread_initial_positions(no_shp, box_length, rad_rep_s, rad_rep_dog, spread_factor=5):
    # Generate positions in the top right corner (positive x, positive y)
    # Using a fixed angle in the top right quadrant
    theta_pos = 2 * np.pi * np.random.rand()
    str_side = box_length * np.array([np.cos(theta_pos), np.sin(theta_pos)])

    # Generate sheep positions with MORE spread (increased from 3 to spread_factor)
    pos_s = str_side - spread_factor * rad_rep_s * np.random.rand(no_shp, 2)

    # Generate dog position with MORE spread
    pos_d = str_side - spread_factor * rad_rep_dog * np.random.rand(2)

    # Generate random initial velocities
    theta_s = 2 * np.pi * np.random.rand(no_shp)
    theta_d = 2 * np.pi * np.random.rand()
    vel_s = np.column_stack([np.cos(theta_s), np.sin(theta_s)])
    vel_d = np.array([np.cos(theta_d), np.sin(theta_d)])

    return pos_s, pos_d, vel_s, vel_d


def main():
    # Simulation parameters
    n_iter = 1000
    num_sheep = 20
    rad_rep_s = 2

    params = dict(
        no_shp=num_sheep,
        box_length=250,
        rad_rep_s=rad_rep_s,
        rad_rep_dog=12,
        K_atr=10,
        k_atr=6,
        k_alg=5,
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

    initial_pos_s, initial_pos_d, initial_vel_s, initial_vel_d = generate_spread_initial_positions(
        no_shp=params['no_shp'],
        box_length=params['box_length'],
        rad_rep_s=params['rad_rep_s'],
        rad_rep_dog=params['rad_rep_dog'],
        spread_factor=10
    )

    pos_s, pos_d, vel_s, vel_d, spd_d, collect_t, drive_t, force_slow_t = herding_model(
        **params,
        initial_pos_s=initial_pos_s,
        initial_pos_d=initial_pos_d,
        initial_vel_s=initial_vel_s,
        initial_vel_d=initial_vel_d
    )

    animation = HerdingAnimation(
        sheep_pos_log=pos_s,
        dog_pos_log=pos_d,
        sheep_vel_log=vel_s,
        dog_vel_log=vel_d,
        dog_speeds_log=spd_d,
        window_size=1200,
        show_metrics=True
    )

    animation.run()

if __name__ == "__main__":
    main()
