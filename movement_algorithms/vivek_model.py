import numpy as np
# Implementation of the Vivek et al. herding model

def herding_model(no_shp, box_length, rad_rep_s, rad_rep_dog, K_atr, k_atr,
                  k_alg, vs, v_dog, h, rho_a, rho_d, e, c, alg_str, f_n,
                  pd, pc, n_iter):
    """

    Args:
        no_shp: Number of sheep (N)
        box_length: Box length
        rad_rep_s: Radius of sheep repulsion (d_Rep)
        rad_rep_dog: Radius of dog repulsion (R_D)
        K_atr: K neighbors that the sheep perceives (k)
        k_atr: k neighbors used for attraction (n_Att, where k_atr <= K_atr?)
        k_alg: k neighbors of alignment (n_Ali)
        vs: Speed of sheep (v_s)
        v_dog: Speed of dog (v_d)
        h: Relative strength of proceeding in the previous direction (inertia) (alpha)
        rho_a: Relative strength of repulsion from neighbors (w_Rep)
        rho_d: Relative strength of repulsion from dog (w_Dog)
        e: Relative strength of angular noise (epsilon)
        c: Relative strength of attraction (w_Att)
        alg_str: Alignment strength (w_Ali)
        f_n: Maximum distance of a sheep from the flock, where it's still considered cohesive(l_Sep)
        pd: Driving position (for calculating l_drive, the distance from Barycenter to P_drive)
        pc: Collecting position (for calculating l_drive, the distance from Barycenter to P_collect)
        n_iter: Number of time steps (iterations)

    Returns:

    """
    # Initialization
    theta_pos = 2 * np.pi * np.random.rand()
    str_side = box_length * np.array([np.cos(theta_pos), np.sin(theta_pos)])
    pos_s = str_side - 3 * rad_rep_s * np.random.rand(no_shp, 2)
    pos_d = str_side - 3 * rad_rep_dog * np.random.rand(2)

    theta_s = 2 * np.pi * np.random.rand(no_shp)
    theta_d = 2 * np.pi * np.random.rand()
    vel_s = np.column_stack([np.cos(theta_s), np.sin(theta_s)])
    vel_d = np.array([np.cos(theta_d), np.sin(theta_d)])
    spd_s = np.ones(no_shp) * vs

    # Storage arrays: (n_iter, no_shp, 2) for easier time-stepping
    pos_s_dat = np.full((n_iter, no_shp, 2), np.nan)
    vel_s_dat = np.full((n_iter, no_shp, 2), np.nan)
    pos_d_dat = np.full((n_iter, 2), np.nan)
    vel_d_dat = np.full((n_iter, 2), np.nan)
    spd_d_dat = np.full(n_iter, np.nan)
    collect_t = np.full(n_iter, np.nan)
    drive_t = np.full(n_iter, np.nan)
    force_slow_t = np.full(n_iter, np.nan)

    # Initial conditions
    pos_s_dat[0] = pos_s
    vel_s_dat[0] = vel_s
    pos_d_dat[0] = pos_d
    vel_d_dat[0] = vel_d
    spd_d_dat[0] = v_dog

    # Time evolution
    pos_s_t_1 = pos_s.copy()
    pos_d_t_1 = pos_d.copy()
    vel_s_t_1 = vel_s.copy()
    vel_d_t_1 = vel_d.copy()

    for t in range(1, n_iter):
        # Sheep movement
        for i in range(no_shp):
            r_shp_dg = pos_d_t_1 - pos_s_t_1[i, :]
            dist_rsd = np.linalg.norm(r_shp_dg)
            r_shp_dg = r_shp_dg / dist_rsd

            if dist_rsd > rad_rep_dog:
                # Beyond dog interaction radius
                r_ij = pos_s_t_1 - pos_s_t_1[i, :]
                mag_rij = np.linalg.norm(r_ij, axis=1)

                rep_j = np.where(mag_rij < rad_rep_s)[0]

                if len(rep_j) > 1:
                    rep_j = rep_j[rep_j != i]
                    r_ij_rep = r_ij[rep_j, :] / mag_rij[rep_j, np.newaxis]
                    r_ij_rep = np.sum(r_ij_rep, axis=0)
                    r_ij_rep = r_ij_rep / np.linalg.norm(r_ij_rep)
                    r_ij_rep = -r_ij_rep

                    vel_next = h * vel_s_t_1[i, :] + rho_a * r_ij_rep
                    vel_next = vel_next / np.linalg.norm(vel_next)

                    pos_s[i, :] = pos_s[i, :] + vs * vel_next
                    vel_s_dat[t, i] = vel_next
                else:
                    vel_s_dat[t, i] = vel_s_t_1[i, :]
            else:
                # Dog is visible
                r_ij = pos_s_t_1 - pos_s_t_1[i, :]
                mag_rij = np.linalg.norm(r_ij, axis=1)

                rep_j = np.where(mag_rij < rad_rep_s)[0]
                is_err = 0
                r_ij_rep = np.zeros(2)

                if len(rep_j) > 1:
                    rep_j = rep_j[rep_j != i]
                    r_ij_rep = r_ij[rep_j, :] / mag_rij[rep_j, np.newaxis]
                    r_ij_rep = np.sum(r_ij_rep, axis=0)
                    r_ij_rep = r_ij_rep / np.linalg.norm(r_ij_rep)
                    r_ij_rep = -r_ij_rep
                    is_err = 1

                # Repulsion from dog
                r_shp_dg = -r_shp_dg

                # Attraction to local center of mass
                lcm_j = np.argsort(mag_rij)[1:K_atr + 1]
                lcm_j = np.random.choice(lcm_j, k_atr, replace=False)
                r_atr = r_ij[lcm_j, :] / (mag_rij[lcm_j, np.newaxis] + np.finfo(float).eps)
                r_atr = np.sum(r_atr, axis=0)
                r_atr = r_atr / np.linalg.norm(r_atr)

                # Alignment
                l_alg = np.random.choice(lcm_j, k_alg, replace=False)
                r_alg = np.sum(vel_s_t_1[l_alg, :], axis=0)
                r_alg = r_alg / np.linalg.norm(r_alg)

                # Random error
                theta_error = 2 * np.pi * np.random.rand()
                r_err = np.array([np.cos(theta_error), np.sin(theta_error)])

                # Resultant velocity
                if is_err == 1:
                    vel_next = (h * vel_s_t_1[i, :] + rho_a * r_ij_rep +
                                rho_d * r_shp_dg + c * r_atr + e * r_err +
                                alg_str * r_alg)
                else:
                    vel_next = (h * vel_s_t_1[i, :] + rho_d * r_shp_dg +
                                c * r_atr + e * r_err + alg_str * r_alg)

                vel_next = vel_next / np.linalg.norm(vel_next)
                pos_s[i, :] = pos_s[i, :] + vs * vel_next
                vel_s_dat[t, i] = vel_next
                spd_s[i] = vs

        # Dog movement
        r_dg_shp = pos_s_t_1 - pos_d_t_1
        dist_rds = np.linalg.norm(r_dg_shp, axis=1)

        if np.min(dist_rds) <= rad_rep_s:
            # Too close to sheep, slow down
            pos_d = pos_d + 0.05 * vel_d_t_1
            vel_d_dat[t] = vel_d_t_1
            spd_d_dat[t] = 0.05
            force_slow_t[t] = 1
        else:
            grp_centre = np.mean(pos_s_t_1, axis=0)
            r_gcm_i = pos_s_t_1 - grp_centre
            dist_gcm_i = np.linalg.norm(r_gcm_i, axis=1)

            if np.max(dist_gcm_i) >= f_n:
                # Collecting behavior
                s_p = np.argmax(dist_gcm_i)
                d_behind = dist_gcm_i[s_p] + pc
                rc = grp_centre + d_behind * (r_gcm_i[s_p, :] / dist_gcm_i[s_p])
                rdc = rc - pos_d_t_1
                rdc = rdc / np.linalg.norm(rdc)

                theta_error = 2 * np.pi * np.random.rand()
                r_err = np.array([np.cos(theta_error), np.sin(theta_error)])

                vel_next = rdc + e * r_err
                vel_next = vel_next / np.linalg.norm(vel_next)

                pos_d = pos_d + v_dog * vel_next
                collect_t[t] = 1
            else:
                # Driving behavior
                # FIXME: The OG code does not take into account a fucking TARGET LMAO WHAT??????????????????
                d_behind = np.linalg.norm(grp_centre) + pd
                r_drive = d_behind * (grp_centre / np.linalg.norm(grp_centre))
                r_drive_orient = r_drive - pos_d_t_1
                r_drive_orient = r_drive_orient / np.linalg.norm(r_drive_orient)

                theta_error = 2 * np.pi * np.random.rand()
                r_err = np.array([np.cos(theta_error), np.sin(theta_error)])

                vel_next = r_drive_orient + e * r_err
                vel_next = vel_next / np.linalg.norm(vel_next)

                pos_d = pos_d + v_dog * vel_next
                drive_t[t] = 1

            vel_d_t_1 = vel_next
            vel_d_dat[t] = vel_next
            spd_d_dat[t] = v_dog

        pos_s_dat[t] = pos_s
        pos_s_t_1 = pos_s.copy()
        vel_s_t_1 = vel_s_dat[t]

        pos_d_dat[t] = pos_d
        pos_d_t_1 = pos_d.copy()

    return (pos_s_dat, pos_d_dat, vel_s_dat, vel_d_dat, spd_d_dat,
            collect_t, drive_t, force_slow_t)