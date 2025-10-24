import numpy as np


# utility
def unit_vector(v):
    """Return the unit vector of v, or zero if norm is 0."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else np.zeros_like(v)


def vecnorm_rows(v):
    """Return Euclidean norm for each row of a 2D array."""
    return np.linalg.norm(v, axis=1)


# Movement functions for each iteration, as modeled in the paper

'''
    ------------------------------ 
        Sheep Movement Functions
    ------------------------------

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
        repulsion_vectors = relative_positions[nearby_sheep_indices] / distances[nearby_sheep_indices]  #[:, None]
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

    # if the dog is far away
    if distance_to_dog > dog_repulsion_radius:
        repulsion_vector = sheep_repulsion_sheep(sheep_index, sheep_positions, sheep_repulsion_radius)
        new_velocity = persistence_weight * sheep_velocities[sheep_index] + sheep_repulsion_weight * repulsion_vector
        new_velocity = unit_vector(new_velocity)
        return new_velocity

    # else, the dog is nearby and is affecting the sheep
    # TODO:


# TODO: maybe delete???
# function that integrates all the drives
def move_sheep(sheep_index, all_sheep_pos, dog_pos):
    shp_pos = all_sheep_pos[sheep_index]
    r_shp_dg = shp_pos - dog_pos
    dist_rsd = r_shp_dg / np.linalg.norm(r_shp_dg)

    raise NotImplementedError
