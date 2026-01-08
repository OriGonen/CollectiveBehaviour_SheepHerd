import numpy as np
# Implementation of the fatigue sheepherding model FSM

# TODO: Do we need TL_Drive and TL_gather? Why are they just equal at all time?
# TODO: Should TL be calculated instead of just using min/max?
# TODO: How well do we track through time, since deltaT=1 for us?

def rc_g_func(M_A, M_F, M_R, TL, epsilon_v):
    RC = M_R + M_A
    return epsilon_v + (1 - epsilon_v) * RC

def dt_g_func(M_A, M_F, M_R, TL, epsilon_v):
    # FIXME: currently this one returns 0 if no danger aka we dont even move away.
    #       Is this correct? I think so TL is zero iff dog is away from animal -> dont move?
    if TL <= 0:
        return 0.0
    return np.clip(M_A / (TL + epsilon_v), 0.0, 1.0)


def herding_model(no_shp, box_length, rad_rep_s, rad_rep_dog, K_atr, k_atr,
                           k_alg, vs, v_dog, h, rho_a, rho_d, e, c, alg_str, f_n,
                           pd, pc, n_iter, delta_t, F_i, R_i, TL_max, TL_chase,
                           L_D, L_R, epsilon_v, TL_gather, TL_drive, TL_idle,
                           initial_state=None,
                           g_func=None):
    if initial_state is None:
        initial_state = {}

    if g_func is None:
        g_func = rc_g_func

    initial_pos_s = initial_state.get('pos_s')
    initial_pos_d = initial_state.get('pos_d')

    if initial_pos_s is None or initial_pos_d is None:
        theta_pos = 2 * np.pi * np.random.rand()
        str_side = box_length * np.array([np.cos(theta_pos), np.sin(theta_pos)])
        pos_s = str_side - 3 * rad_rep_s * np.random.rand(no_shp, 2)
        pos_d = str_side - 3 * rad_rep_dog * np.random.rand(2)
    else:
        pos_s = initial_pos_s.copy()
        pos_d = initial_pos_d.copy()

    initial_vel_s = initial_state.get('vel_s')
    initial_vel_d = initial_state.get('vel_d')

    if initial_vel_s is None or initial_vel_d is None:
        theta_s = 2 * np.pi * np.random.rand(no_shp)
        theta_d = 2 * np.pi * np.random.rand()
        vel_s = np.column_stack([np.cos(theta_s), np.sin(theta_s)])
        vel_d = np.array([np.cos(theta_d), np.sin(theta_d)])
    else:
        vel_s = initial_vel_s.copy()
        vel_d = initial_vel_d.copy()

    # Initial fatigue states
    initial_M_A_s = initial_state.get('M_A_s')
    if initial_M_A_s is None:
        M_A_s = np.zeros(no_shp)
    else:
        M_A_s = initial_M_A_s.copy()

    initial_M_F_s = initial_state.get('M_F_s')
    if initial_M_F_s is None:
        M_F_s = np.zeros(no_shp)
    else:
        M_F_s = initial_M_F_s.copy()

    M_R_s = 1 - M_A_s - M_F_s

    initial_M_A_d = initial_state.get('M_A_d')
    if initial_M_A_d is None:
        M_A_d = 0.0
    else:
        M_A_d = initial_M_A_d

    initial_M_F_d = initial_state.get('M_F_d')
    if initial_M_F_d is None:
        M_F_d = 0.0
    else:
        M_F_d = initial_M_F_d

    M_R_d = 1 - M_A_d - M_F_d

    #TODO: I think this needs to be changed aswell?
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

    # Fatigue storage
    M_A_s_dat = np.full((n_iter, no_shp), np.nan)
    M_F_s_dat = np.full((n_iter, no_shp), np.nan)
    M_R_s_dat = np.full((n_iter, no_shp), np.nan)
    TL_s_dat = np.full((n_iter, no_shp), np.nan)
    C_s_dat = np.full((n_iter, no_shp), np.nan)

    M_A_d_dat = np.full(n_iter, np.nan)
    M_F_d_dat = np.full(n_iter, np.nan)
    M_R_d_dat = np.full(n_iter, np.nan)
    TL_d_dat = np.full(n_iter, np.nan)
    C_d_dat = np.full(n_iter, np.nan)

    # Initial conditions
    pos_s_dat[0] = pos_s
    vel_s_dat[0] = vel_s
    pos_d_dat[0] = pos_d
    vel_d_dat[0] = vel_d
    spd_d_dat[0] = v_dog

    M_A_s_dat[0] = M_A_s
    M_F_s_dat[0] = M_F_s
    M_R_s_dat[0] = M_R_s
    M_A_d_dat[0] = M_A_d
    M_F_d_dat[0] = M_F_d
    M_R_d_dat[0] = M_R_d

    # Time evolution
    pos_s_t_1 = pos_s.copy()
    pos_d_t_1 = pos_d.copy()
    vel_s_t_1 = vel_s.copy()
    vel_d_t_1 = vel_d.copy()

    for t in range(1, n_iter):
        # Update fatigue states for sheep
        for i in range(no_shp):
            # FIXME: Maybe flip the - just to be the same as below
            dist_iD = np.linalg.norm(pos_s_t_1[i, :] - pos_d_t_1)
            if dist_iD >= rad_rep_dog:
                TL_i = 0.0
            else:
                TL_i = TL_max * np.clip(1 - dist_iD / rad_rep_dog, 0, 1)

            if M_A_s[i] < TL_i:
                if M_R_s[i] >= (TL_i - M_A_s[i]):
                    C_i = L_D * (TL_i - M_A_s[i])
                else:
                    C_i = L_D * M_R_s[i]
            else:
                C_i = L_R * (TL_i - M_A_s[i])

            M_R_next = M_R_s[i] + delta_t * (-C_i + R_i * M_F_s[i])
            M_A_next = M_A_s[i] + delta_t * (C_i - F_i * M_A_s[i])
            M_F_next = M_F_s[i] + delta_t * (F_i * M_A_s[i] - R_i * M_F_s[i])

            M_A_s[i] = np.clip(M_A_next, 0, 1)
            M_F_s[i] = np.clip(M_F_next, 0, 1)
            M_R_s[i] = 1 - M_A_s[i] - M_F_s[i]
            # Clipping M_R and re-adjusting is handled by taking it from simplex
            # FIXME: This should not be needed
            if M_R_s[i] < 0:
                M_R_s[i] = 0
                # Re-normalize if needed, though clip(0,1) and M_R=1-A-F should mostly work
                total = M_A_s[i] + M_F_s[i]
                if total > 1:
                    M_A_s[i] /= total
                    M_F_s[i] /= total
                    M_R_s[i] = 0
            elif M_R_s[i] > 1:
                M_R_s[i] = 1
                M_A_s[i] = 0
                M_F_s[i] = 0

            TL_s_dat[t, i] = TL_i
            C_s_dat[t, i] = C_i

        M_A_s_dat[t] = M_A_s
        M_F_s_dat[t] = M_F_s
        M_R_s_dat[t] = M_R_s

        # Sheep movement
        for i in range(no_shp):
            v_i_eff = vs * g_func(M_A_s[i], M_F_s[i], M_R_s[i], TL_s_dat[t, i], epsilon_v)
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

                    pos_s[i, :] = pos_s[i, :] + v_i_eff * vel_next
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
                pos_s[i, :] = pos_s[i, :] + v_i_eff * vel_next
                vel_s_dat[t, i] = vel_next
                spd_s[i] = v_i_eff

        # Dog movement logic to determine TL_d
        r_dg_shp = pos_s_t_1 - pos_d_t_1
        dist_rds = np.linalg.norm(r_dg_shp, axis=1)

        if np.min(dist_rds) <= rad_rep_s:
            # FIXME: shouldnt this still be setting to 0.05?
            TL_d = TL_idle
            mode = "slow"
        else:
            grp_centre = np.mean(pos_s_t_1, axis=0)
            r_gcm_i = pos_s_t_1 - grp_centre
            dist_gcm_i = np.linalg.norm(r_gcm_i, axis=1)

            if np.max(dist_gcm_i) >= f_n:
                TL_d = TL_gather
                mode = "collect"
            else:
                TL_d = TL_drive
                mode = "drive"

        # Update dog fatigue state
        if M_A_d < TL_d:
            if M_R_d >= (TL_d - M_A_d):
                C_d = L_D * (TL_d - M_A_d)
            else:
                C_d = L_D * M_R_d
        else:
            C_d = L_R * (TL_d - M_A_d)

        M_R_next_d = M_R_d + delta_t * (-C_d + R_i * M_F_d)
        M_A_next_d = M_A_d + delta_t * (C_d - F_i * M_A_d)
        M_F_next_d = M_F_d + delta_t * (F_i * M_A_d - R_i * M_F_d)

        M_A_d = np.clip(M_A_next_d, 0, 1)
        M_F_d = np.clip(M_F_next_d, 0, 1)
        M_R_d = 1 - M_A_d - M_F_d
        # FIXME: should not be needed
        if M_R_d < 0:
            M_R_d = 0
            total = M_A_d + M_F_d
            if total > 1:
                M_A_d /= total
                M_F_d /= total
                M_R_d = 0
        elif M_R_d > 1:
            M_R_d = 1
            M_A_d = 0
            M_F_d = 0

        TL_d_dat[t] = TL_d
        C_d_dat[t] = C_d
        M_A_d_dat[t] = M_A_d
        M_F_d_dat[t] = M_F_d
        M_R_d_dat[t] = M_R_d

        v_d_eff = v_dog * g_func(M_A_d, M_F_d, M_R_d, TL_d, epsilon_v)

        # Dog movement
        if mode == "slow":
            # Too close to sheep, slow down
            pos_d = pos_d + 0.05 * v_d_eff * vel_d_t_1
            vel_d_dat[t] = vel_d_t_1
            spd_d_dat[t] = 0.05 * v_d_eff
            force_slow_t[t] = 1
        elif mode == "collect":
            # Collecting behavior
            grp_centre = np.mean(pos_s_t_1, axis=0)
            r_gcm_i = pos_s_t_1 - grp_centre
            dist_gcm_i = np.linalg.norm(r_gcm_i, axis=1)
            s_p = np.argmax(dist_gcm_i)
            d_behind = dist_gcm_i[s_p] + pc
            rc = grp_centre + d_behind * (r_gcm_i[s_p, :] / dist_gcm_i[s_p])
            rdc = rc - pos_d_t_1
            rdc = rdc / np.linalg.norm(rdc)

            theta_error = 2 * np.pi * np.random.rand()
            r_err = np.array([np.cos(theta_error), np.sin(theta_error)])

            vel_next = rdc + e * r_err
            vel_next = vel_next / np.linalg.norm(vel_next)

            pos_d = pos_d + v_d_eff * vel_next
            collect_t[t] = 1
            vel_d_t_1 = vel_next
            vel_d_dat[t] = vel_next
            spd_d_dat[t] = v_d_eff
        else: # drive
            # Driving behavior
            grp_centre = np.mean(pos_s_t_1, axis=0)
            d_behind = np.linalg.norm(grp_centre) + pd
            r_drive = d_behind * (grp_centre / np.linalg.norm(grp_centre))
            r_drive_orient = r_drive - pos_d_t_1
            r_drive_orient = r_drive_orient / np.linalg.norm(r_drive_orient)

            theta_error = 2 * np.pi * np.random.rand()
            r_err = np.array([np.cos(theta_error), np.sin(theta_error)])

            vel_next = r_drive_orient + e * r_err
            vel_next = vel_next / np.linalg.norm(vel_next)

            pos_d = pos_d + v_d_eff * vel_next
            drive_t[t] = 1
            vel_d_t_1 = vel_next
            vel_d_dat[t] = vel_next
            spd_d_dat[t] = v_d_eff

        pos_s_dat[t] = pos_s
        pos_s_t_1 = pos_s.copy()
        vel_s_t_1 = vel_s_dat[t]

        pos_d_dat[t] = pos_d
        pos_d_t_1 = pos_d.copy()

    return {
        "pos_s_dat": pos_s_dat,
        "pos_d_dat": pos_d_dat,
        "vel_s_dat": vel_s_dat,
        "vel_d_dat": vel_d_dat,
        "spd_d_dat": spd_d_dat,
        "collect_t": collect_t,
        "drive_t": drive_t,
        "force_slow_t": force_slow_t,
        "M_A_s_dat": M_A_s_dat,
        "M_F_s_dat": M_F_s_dat,
        "M_R_s_dat": M_R_s_dat,
        "TL_s_dat": TL_s_dat,
        "C_s_dat": C_s_dat,
        "M_A_d_dat": M_A_d_dat,
        "M_F_d_dat": M_F_d_dat,
        "M_R_d_dat": M_R_d_dat,
        "TL_d_dat": TL_d_dat,
        "C_d_dat": C_d_dat
    }