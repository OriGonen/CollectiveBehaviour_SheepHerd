from Movement_Algorithms.fatigue_model import herding_model

import numpy as np

from animation import HerdingAnimation

if __name__ == "__main__":
    num_sheep = 14

    rad_rep_s = 2

    params = dict(
        no_shp=num_sheep,
        box_length=500,
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
        n_iter=300,

    )

    black_sheep_size = 0
    white_sheep_size = 1 - black_sheep_size


    ones = np.ones(int(num_sheep/2))/2
    zeros = np.zeros(int(num_sheep/2))+0.1
    #print(np.full(num_sheep, 1))
    F_i = np.concatenate([ones,zeros])
    R_i = np.concatenate([zeros,zeros])

    #F_i = np.ones(num_sheep)
    #R_i = np.ones(num_sheep)
    F_d = 0.5
    R_d = 0.1
    L_D=1
    results = herding_model(**params, F_i=F_i, R_i=R_i, L_D=L_D, L_R=L_D, v_d_close=0.05,
                            F_d = F_d, R_d = R_d,L_R_d=9, L_D_d=9,
                            TL_max_dog=1, TL_max_soc=0.1, TL_gather=1, v_s_min=0.1, v_d_min=0.1, TL_drive=1)

    anim = HerdingAnimation(results['pos_s_dat'], results['pos_d_dat'],
                            results['vel_s_dat'], results['vel_d_dat'],
                            dog_speeds_log=results['spd_d_dat'], show_metrics=True)
    anim.run()