import sys

import numpy as np

# -----------------------
# Utility functions
# -----------------------
EPS = 1e-8  # epsilon, for approximating little things

def unit_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else np.zeros_like(v)

def random_unit_vector():
    angle = np.random.uniform(-np.pi, np.pi)
    return np.array([np.cos(angle), np.sin(angle)])

def vecnorm_rows(v):
    return np.linalg.norm(v, axis=1)


# -----------------------
#   Sheep functions
# -----------------------
def sheep_repulsion_sheep(sheep_index, all_sheep_pos, repulsion_range):
    relative_positions = all_sheep_pos - all_sheep_pos[sheep_index]  # Aj - Ai
    distances = vecnorm_rows(relative_positions)
    nearby = np.where((distances < repulsion_range) & (distances > 0))[0]  # exclude self

    if len(nearby) == 0:
        return np.zeros(2)

    # calculate
    dirs = []
    for idx in nearby:
        d = distances[idx]
        rel = relative_positions[idx]  # Aj - Ai
        if d > EPS:
            dirs.append(-rel / d)  # point away from neighbor
        else:
            dirs.append(random_unit_vector())
    unit_dirs = np.vstack(dirs)
    Ra_sum = np.sum(unit_dirs, axis=0)
    return Ra_sum


def sheep_attraction_center(sheep_index, all_sheep_pos, n_neighbors):
    relative_positions = all_sheep_pos - all_sheep_pos[sheep_index]
    distances = vecnorm_rows(relative_positions)
    # sort and pick n nearest neighbors, excluding self
    nearest = np.argsort(distances)[1:n_neighbors + 1]
    if len(nearest) == 0:
        return np.zeros(2)
    centroid = np.mean(all_sheep_pos[nearest], axis=0)
    return centroid - all_sheep_pos[sheep_index]


# -----------------------
# dog functions
# -----------------------
def dog_collecting_point(sheep_positions, group_center, furthest_index, ra_dist, collecting_offset):
    Af = sheep_positions[furthest_index]
    vec = Af - group_center
    vec_unit = unit_vector(vec)
    if np.all(vec_unit == 0):
        vec_unit = random_unit_vector()
    Pc = Af + (ra_dist + collecting_offset) * vec_unit
    return Pc


def dog_driving_point(group_center, goal, ra_dist, N, driving_offset):
    offset = ra_dist * np.sqrt(N) + driving_offset
    vec = group_center - goal
    vec_unit = unit_vector(vec)
    if np.all(vec_unit == 0):
        vec_unit = random_unit_vector()
    Pd = group_center + offset * vec_unit
    return Pd


def update_dog_position_strombom(sheep_positions, dog_position,
                                 ra_dist, goal, N,
                                 e_noise, ds, collecting_offset=0.0, driving_offset=0.0):

    dists = vecnorm_rows(sheep_positions - dog_position)
    min_dist = np.min(dists)

    group_center = np.mean(sheep_positions, axis=0)
    distances_to_center = vecnorm_rows(sheep_positions - group_center)
    fN = ra_dist * (N ** 0.5)

    if min_dist <= 3.0 * ra_dist:
        noisy_heading = random_unit_vector()
        return dog_position.copy(), noisy_heading, "stop"

    # modes of dog
    if np.max(distances_to_center) <= fN:
        target = dog_driving_point(group_center, goal, ra_dist, N, driving_offset)
        mode = "drive"
    else:
        furthest_idx = int(np.argmax(distances_to_center))
        target = dog_collecting_point(sheep_positions, group_center, furthest_idx, ra_dist, collecting_offset)
        mode = "collect"


    direction = target - dog_position
    # check if target too close
    dist_to_target = np.linalg.norm(direction)
    if dist_to_target < EPS:
        # tiny random move to change heading
        dir_unit = random_unit_vector()
        move = 0.0
        new_position = dog_position.copy()
    else:
        direction_noisy = direction + e_noise * random_unit_vector()
        dir_unit = unit_vector(direction_noisy)
        move = min(ds, dist_to_target)   # avoid overshoot
        new_position = dog_position + move * dir_unit

    return new_position, dir_unit, mode



# -----------------------
# Simulation Strombom
# -----------------------
def simulate_model_strombom_main(
        num_sheep, box_length,
        ra_dist, rs_range,        # distances
        n_neighbors,             # number of neighbors for attraction
        d_step, ds,              # step sizes
        h_weight, c_weight, ra_weight, rs_weight,  # heading weights
        e_noise, p_move,         # noise and grazing prob
        goal=(0.0, 0.0),
        collecting_offset=0.0, driving_offset=0.0,
        num_iterations=1000
):
    # initialization
    random_angle = 2 * np.pi * np.random.rand()
    start_side = box_length * np.array([np.cos(random_angle), np.sin(random_angle)])
    sheep_positions = start_side - 3 * ra_dist * np.random.rand(num_sheep, 2)
    dog_position = (start_side - 3 * rs_range * np.random.rand(1, 2)).flatten()

    sheep_angles = 2 * np.pi * np.random.rand(num_sheep)
    dog_angle = 2 * np.pi * np.random.rand()
    sheep_velocities = np.column_stack((np.cos(sheep_angles), np.sin(sheep_angles)))  # previous headings
    dog_velocity = np.array([np.cos(dog_angle), np.sin(dog_angle)])

    sheep_positions_log = np.full((num_iterations, num_sheep, 2), np.nan)
    sheep_velocities_log = np.full((num_iterations, num_sheep, 2), np.nan)
    dog_positions_log = np.full((num_iterations, 2), np.nan)
    dog_velocities_log = np.full((num_iterations, 2), np.nan)
    dog_speeds_log = np.full((num_iterations,), np.nan)
    collecting_flags = np.zeros(num_iterations)
    driving_flags = np.zeros(num_iterations)
    stopping_flags = np.zeros(num_iterations)

    # initialization
    sheep_positions_log[0] = sheep_positions
    sheep_velocities_log[0] = sheep_velocities
    dog_positions_log[0] = dog_position
    dog_velocities_log[0] = dog_velocity
    dog_speeds_log[0] = 0.0

    for t in range(1, num_iterations):

        # print percentage progress
        sys.stdout.write(f"\r{(t+1)/num_iterations*100:6.2f} %")


        # compute headings
        new_headings = np.zeros_like(sheep_velocities)
        for i in range(num_sheep):
            dir_to_dog = dog_position - sheep_positions[i]
            dist_to_dog = np.linalg.norm(dir_to_dog)

            if dist_to_dog > rs_range:
                new_headings[i] = unit_vector(sheep_velocities[i])  # grazing
            else:
                # repulsion (sum of Ai - Aj)
                relative_positions = sheep_positions - sheep_positions[i]  # Aj - Ai
                distances = vecnorm_rows(relative_positions)
                nearby_idxs = np.where((distances < ra_dist) & (distances > 0))[0]
                if len(nearby_idxs) > 0:
                    unit_dirs = np.zeros((len(nearby_idxs), 2))
                    for k, idx in enumerate(nearby_idxs):
                        d = distances[idx]
                        rel = relative_positions[idx]
                        if d > EPS:
                            unit_dirs[k] = -rel / d  # NEGATE here: (Ai - Aj)/|Ai-Aj|
                        else:
                            unit_dirs[k] = random_unit_vector()
                    Ra_vec = np.sum(unit_dirs, axis=0)
                else:
                    Ra_vec = np.zeros(2)

                C_vec = sheep_attraction_center(i, sheep_positions, n_neighbors)
                C_hat = unit_vector(C_vec)
                H_prev = unit_vector(sheep_velocities[i])
                Rs_hat = unit_vector(sheep_positions[i] - dog_position)  # away from dog
                eta_hat = random_unit_vector()

                # H'
                Hprime = (h_weight * H_prev
                          + c_weight * C_hat
                          + ra_weight * Ra_vec
                          + rs_weight * Rs_hat
                          + e_noise * eta_hat)
                new_headings[i] = unit_vector(Hprime)

        # apply grazing moves and update positions synchronously
        new_positions = sheep_positions.copy()
        for i in range(num_sheep):
            dir_to_dog = dog_position - sheep_positions[i]
            dist_to_dog = np.linalg.norm(dir_to_dog)
            if dist_to_dog > rs_range:
                if np.random.rand() < p_move:
                    rdir = random_unit_vector()
                    new_positions[i] = sheep_positions[i] + d_step * rdir
                    new_headings[i] = unit_vector(rdir)
                else:
                    new_positions[i] = sheep_positions[i]
            else:
                new_positions[i] = sheep_positions[i] + d_step * new_headings[i]

        sheep_positions = new_positions
        sheep_velocities = new_headings

        # dog update
        new_dog_pos, dog_heading, mode = update_dog_position_strombom(
            sheep_positions, dog_position,
            ra_dist, np.array(goal), num_sheep,
            e_noise, ds, collecting_offset, driving_offset
        )


        dog_position = new_dog_pos
        dog_velocity = dog_heading

        current_speed = 0.0
        if mode != "stop":
            current_speed = np.linalg.norm(dog_heading) * ds

        if mode == "collect":
            collecting_flags[t] = 1
        elif mode == "drive":
            driving_flags[t] = 1
        elif mode == "stop":
            stopping_flags[t] = 1
            current_speed = 0.0

        # store
        sheep_positions_log[t] = sheep_positions
        sheep_velocities_log[t] = sheep_velocities
        dog_positions_log[t] = dog_position
        dog_velocities_log[t] = dog_velocity
        dog_speeds_log[t] = current_speed

    return (sheep_positions_log, dog_positions_log,
            sheep_velocities_log, dog_velocities_log,
            dog_speeds_log, collecting_flags, driving_flags, stopping_flags)
