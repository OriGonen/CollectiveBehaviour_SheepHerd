import sys

import numpy as np


# utility
def unit_vector(v):
    # return normalized v
    norm = np.linalg.norm(v)
    # if v is the zero vector then return zero vector
    return v / norm if norm > 0 else np.zeros_like(v)


def random_unit_vector():
    # samples a random unit vector
    angle = 2 * np.pi * np.random.rand()
    return np.array([np.cos(angle), np.sin(angle)])


def vecnorm_rows(v):
    """Return Euclidean norm for each row of a 2D array."""
    return np.linalg.norm(v, axis=1)


# below are the movement functions for each iteration, as modeled in the paper
'''
    -------------------------------- 
        Sheep Movement Functions
    --------------------------------

The drives of the sheep:
1. Repulsion from other sheep
2. Attraction to center of local neighbors
3. Velocity direction alignment to average of neighbors
4. Repulsion from dog
5. Random noise to the movement
'''


# Repulsion from other sheep
def sheep_repulsion_sheep(sheep_index, all_sheep_pos, sheep_repulsion_radius):
    # returns a unit vector away from nearby sheep or zero vector if none are nearby
    relative_positions = all_sheep_pos - all_sheep_pos[sheep_index]
    distances = vecnorm_rows(relative_positions)
    nearby_sheep_indices = np.where(distances < sheep_repulsion_radius)[0]
    nearby_sheep_indices = nearby_sheep_indices[nearby_sheep_indices != sheep_index]

    if len(nearby_sheep_indices) > 0:
        repulsion_vectors = relative_positions[nearby_sheep_indices] / distances[nearby_sheep_indices][:, None]
        total_repulsion = -unit_vector(np.sum(repulsion_vectors, axis=0))
        return total_repulsion

    else:
        # no sheep nearby, return zeros
        return np.zeros(2)


# Attraction to center of local neighbors
def sheep_attraction_center(sheep_index, all_sheep_pos, num_neighbors_for_attraction, num_random_neighbors):
    # the sheep will observe the num_neighbors_for_attraction closest neighbors, pick at random num_random_neighbors
    # of them and be attracted towards their centroid
    relative_positions = all_sheep_pos - all_sheep_pos[sheep_index]
    distances = vecnorm_rows(relative_positions)
    # we start from 1 and end on +1 to exclude the sheep itself from being a neighbor
    nearest = np.argsort(distances)[1:num_neighbors_for_attraction + 1]
    # pick random neighbors (without repetitions!)
    chosen_nearest = np.random.choice(nearest, num_random_neighbors, replace=False)
    # (we add the 1e-8 to protect against zero divisions)
    attraction_vector = np.sum(relative_positions[chosen_nearest] /
                               (distances[chosen_nearest][:, None] + 1e-8), axis=0)
    # normalize and return
    return unit_vector(attraction_vector)


# Velocity direction alignment to average of neighbors
def sheep_alignment(sheep_velocities, neighbor_indices, num_alignment_neighbors):
    # align velocity in accordance to random neighbors
    chosen_neighbors = np.random.choice(neighbor_indices, num_alignment_neighbors, replace=False)
    average_direction = np.sum(sheep_velocities[chosen_neighbors], axis=0)
    return unit_vector(average_direction)


# compute the next velocity for a sheep given the influences
# repulsion from other sheep and the dog, attraction to center, velocity alignment, and noise
def update_sheep_velocity(
        sheep_positions, sheep_velocities, dog_position, sheep_index, sheep_repulsion_radius, dog_repulsion_radius,
        num_neighbors_for_attraction, num_random_neighbors, num_alignment_neighbors, persistence_weight,
        sheep_repulsion_weight, dog_repulsion_weight, attraction_weight, noise_weight, alignment_weight
):
    direction_to_dog = dog_position - sheep_positions[sheep_index]
    distance_to_dog = np.linalg.norm(direction_to_dog)
    direction_to_dog /= distance_to_dog if distance_to_dog > 0 else 1

    repulsion_vector = sheep_repulsion_sheep(sheep_index, sheep_positions, sheep_repulsion_radius)

    # case: the dog is far away
    # in this case, the sheep do not align and do not attract towards the center
    if distance_to_dog > dog_repulsion_radius:
        new_velocity = persistence_weight * sheep_velocities[sheep_index] + sheep_repulsion_weight * repulsion_vector
        new_velocity = unit_vector(new_velocity)
        return new_velocity

    # else, the dog is nearby and is affecting the sheep
    dog_avoidance_vector = -direction_to_dog
    attraction_vector = sheep_attraction_center(sheep_index, sheep_positions,
                                                num_neighbors_for_attraction, num_random_neighbors)

    sheep_relative_positions = sheep_positions - sheep_positions[sheep_index]
    sheep_distances = vecnorm_rows(sheep_relative_positions)
    sorted_neighbor_indices = np.argsort(sheep_distances)
    # start from 1 to exclude self
    attraction_neighbors = sorted_neighbor_indices[1:num_neighbors_for_attraction + 1]
    alignment_vector = sheep_alignment(sheep_velocities, attraction_neighbors, num_alignment_neighbors)

    # add random noise
    noise_vector = random_unit_vector()

    # now, combine all vectors
    updated_velocity = (persistence_weight * sheep_velocities[sheep_index]
                        + sheep_repulsion_weight * repulsion_vector
                        + dog_repulsion_weight * dog_avoidance_vector
                        + attraction_weight * attraction_vector
                        + alignment_weight * alignment_vector
                        + noise_weight * noise_vector
                        )
    return unit_vector(updated_velocity)


'''
    ------------------------------
        Dog Movement Functions
    ------------------------------
    
    The dog has two modes of operation
    1. Collecting: the dog moves to collect stray sheep
            This happens when the group is not cohesive.
    2. Driving: the dog moves behind the herd to drive it forward
            This happens when the group is cohesive.
            
'''


# the dog moves to collect stray sheep when the group is not cohesive
def dog_collecting_mode(sheep_positions, dog_position, group_center, non_cohesive_distance,
                        noise_weight, dog_speed, collecting_offset):
    relative_positions_to_center = sheep_positions - group_center
    distances_to_center = vecnorm_rows(relative_positions_to_center)

    # find the stray-est sheep
    stray_sheep_index = np.argmax(distances_to_center)
    behind_distance = distances_to_center[stray_sheep_index] + collecting_offset
    target_position = group_center + behind_distance * (relative_positions_to_center[stray_sheep_index] / distances_to_center[stray_sheep_index])
    direction_to_target = unit_vector(target_position - dog_position)
    next_velocity = unit_vector(direction_to_target + noise_weight * random_unit_vector())

    # compute next dog position
    next_position = dog_position + dog_speed * next_velocity
    return next_position, next_velocity, "collect"


# the dog moves behind the group and drives it forward when the herd is cohesive
def dog_driving_mode(sheep_positions, dog_position, group_center, noise_weight, dog_speed, driving_offset):
    # calculate the direction the dog moves to
    behind_distance = np.linalg.norm(group_center) + driving_offset
    desired_drive_position = behind_distance * unit_vector(group_center)
    drive_direction = unit_vector(desired_drive_position - dog_position)
    # add some random noise to next velocity direction
    next_velocity = unit_vector(drive_direction + noise_weight * random_unit_vector())

    next_position = dog_position + dog_speed * next_velocity
    return next_position, next_velocity, "drive"


# compute the next movement of the dog given the sheep positions and other influences
def update_dog_movement(sheep_positions, dog_position, sheep_repulsion_radius, non_cohesive_distance,
                        driving_offset, collecting_offset, noise_weight, dog_speed):
    # check if dog is too close to the sheep
    distances_to_sheep = vecnorm_rows(sheep_positions - dog_position)
    if np.min(distances_to_sheep) <= sheep_repulsion_radius:
        # dog is too close, need to slow down
        braking_factor = 0.05  # greatly reduce the speed for the next step to not run into the sheep and scatter them
        # instead of moving forward, move slowly to another direction
        new_direction = unit_vector(np.random.randn(2))
        new_position = dog_position + braking_factor * dog_speed * new_direction
        return new_position, braking_factor*dog_speed*new_direction, "slow"

    # else, there's no need to brake and the dog moves in accordance to cohesiveness of the herd
    group_center = np.mean(sheep_positions, axis=0)
    relative_positions_to_center = sheep_positions - group_center
    distances_to_center = vecnorm_rows(relative_positions_to_center)

    # check if the most strayed sheep is farther from the threshold for cohesiveness
    if np.max(distances_to_center) >= non_cohesive_distance:
        # the herd is not cohesive - the dog needs to collect
        return dog_collecting_mode(sheep_positions, dog_position, group_center, non_cohesive_distance,
                                   noise_weight, dog_speed, collecting_offset)
    else:
        # the group is cohesive - the dog can drive the herd
        return dog_driving_mode(sheep_positions, dog_position, group_center, noise_weight, dog_speed, driving_offset)



'''
    ----------------------------
        simulation function
    ----------------------------
    the simulation function utilizes the movement functions to simulate the behaviour of the sheep and dog
    the function will return arrays for positions, velocities, and dog-modes for each timestep.
'''


def simulate_model(
        num_sheep, box_length,
        sheep_repulsion_radius, dog_repulsion_radius,
        num_neighbors_for_attraction, num_random_attraction_neighbors, num_alignment_neighbors,
        sheep_speed, dog_speed, persistence_weight,
        sheep_repulsion_weight, dog_repulsion_weight,
        noise_weight, attraction_weight, alignment_weight,
        non_cohesive_distance, driving_offset, collecting_offset, num_iterations
):
    # --- Initialization ---
    random_angle = 2 * np.pi * np.random.rand()
    start_side = box_length * np.array([np.cos(random_angle), np.sin(random_angle)])
    sheep_positions = start_side - 3 * sheep_repulsion_radius * np.random.rand(num_sheep, 2)
    dog_position = (start_side - 3 * dog_repulsion_radius * np.random.rand(1, 2)).flatten()

    sheep_angles = 2 * np.pi * np.random.rand(num_sheep)
    dog_angle = 2 * np.pi * np.random.rand()
    sheep_velocities = np.column_stack((np.cos(sheep_angles), np.sin(sheep_angles)))
    dog_velocity = np.array([np.cos(dog_angle), np.sin(dog_angle)])

    # --- Data storage ---
    sheep_positions_log = np.full((num_iterations, num_sheep, 2), np.nan)
    sheep_velocities_log = np.full((num_iterations, num_sheep, 2), np.nan)
    dog_positions_log = np.full((num_iterations, 2), np.nan)
    dog_velocities_log = np.full((num_iterations, 2), np.nan)
    dog_speeds_log = np.full((num_iterations,), np.nan)
    collecting_flags = np.zeros(num_iterations)
    driving_flags = np.zeros(num_iterations)
    slowing_flags = np.zeros(num_iterations)

    # Store initial state
    sheep_positions_log[0] = sheep_positions
    sheep_velocities_log[0] = sheep_velocities
    dog_positions_log[0] = dog_position
    dog_velocities_log[0] = dog_velocity
    dog_speeds_log[0] = dog_speed

    # --- Simulation loop ---
    for t in range(1, num_iterations):

        # print percentage progress
        sys.stdout.write(f"\r{(t + 1) / num_iterations * 100:6.2f} %")

        # Sheep movement
        for i in range(num_sheep):
            new_velocity = update_sheep_velocity(
                sheep_positions, sheep_velocities, dog_position, i,
                sheep_repulsion_radius, dog_repulsion_radius,
                num_neighbors_for_attraction, num_random_attraction_neighbors, num_alignment_neighbors,
                persistence_weight, sheep_repulsion_weight, dog_repulsion_weight,
                attraction_weight, noise_weight, alignment_weight
            )
            sheep_positions[i] += sheep_speed * new_velocity
            sheep_velocities[i] = new_velocity

        # Dog movement
        dog_position, dog_velocity, mode = update_dog_movement(
            sheep_positions, dog_position, sheep_repulsion_radius,
            non_cohesive_distance, driving_offset, collecting_offset,
            noise_weight, dog_speed
        )

        current_speed = dog_speed

        if mode == "collect":
            collecting_flags[t] = 1
        elif mode == "drive":
            driving_flags[t] = 1
        elif mode == "slow":
            slowing_flags[t] = 1
            braking_factor = 0.05
            current_speed = braking_factor * dog_speed  # braking if too close

        # Store time step data
        sheep_positions_log[t] = sheep_positions
        sheep_velocities_log[t] = sheep_velocities
        dog_positions_log[t] = dog_position
        dog_velocities_log[t] = dog_velocity
        dog_speeds_log[t] = current_speed

    return (sheep_positions_log, dog_positions_log,
            sheep_velocities_log, dog_velocities_log,
            dog_speeds_log, collecting_flags, driving_flags, slowing_flags)
