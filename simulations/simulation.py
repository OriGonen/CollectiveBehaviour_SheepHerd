from movement_algorithms.vivek_model import herding_model

import numpy as np

from animation import HerdingAnimation

# TODO:
# - Add animation with barycenter arrows + individuals
# - Add animation with "trail"
# - Test UI

if __name__ == "__main__":
    num_sheep = 14

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
        f_n=rad_rep_s * (num_sheep ** (2/3)),
        pd=rad_rep_s * np.sqrt(num_sheep),
        pc=rad_rep_s,
        n_iter=370
    )

    sheep_positions_log, dog_positions_log, \
        sheep_velocities_log, dog_velocities_log, \
        dog_speeds_log, collecting_flags, driving_flags, slowing_flags = herding_model(**params)

    anim = HerdingAnimation(sheep_positions_log, dog_positions_log,
                            sheep_velocities_log, dog_velocities_log,
                            dog_speeds_log=dog_speeds_log, show_metrics=True)
    anim.run()