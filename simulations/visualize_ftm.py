from movement_algorithms.fatigue_model import herding_model

import numpy as np

from animation import HerdingAnimation

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
        n_iter=370,

    )

    F_i = np.full(num_sheep, 0.1)
    R_i = np.full(num_sheep, 0.02)
    L_D=10
    results = herding_model(**params, F_i=F_i, R_i=R_i, L_D=L_D, L_R=L_D, v_d_close=0.05,
                            F_d = 0.05, R_d = 0.05,L_R_d=9, L_D_d=9,
                            TL_max_dog=1, TL_max_soc=0.1, TL_gather=1, v_s_min=0.1, v_d_min=0.1, TL_drive=1)

    anim = HerdingAnimation(results['pos_s_dat'], results['pos_d_dat'],
                            results['vel_s_dat'], results['vel_d_dat'],
                            dog_speeds_log=results['spd_d_dat'], show_metrics=True)
    anim.run()