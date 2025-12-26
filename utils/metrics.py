import numpy as np


# Metrics calculations are sourced from the paper: Quantification of collective behavior (page 8).

def calculate_cohesion_sim(sheep_positions_simulation):
    """
    Calculate cohesion for sheep flock simulations.

    Cohesion is defined as the average distance of all sheep from the
    flock's barycenter at each time step.

    Mathematical definition:
        C(t) = (1/N) * sum(||r_i(t) - R(t)||)
        where R(t) = (1/N) * sum(r_i(t)) is the barycenter

    Parameters:
    -----------
    sheep_positions_simulation : np.ndarray
        Array of shape (no_it, n_iter, no_shp, 2) containing position vectors
        where the last axis contains (x, y) coordinates.

    Returns:
    --------
    np.ndarray
        Cohesion values of shape (no_it, n_iter)
    """
    r_b = np.mean(sheep_positions_simulation, axis=2, keepdims=True)
    distances = np.linalg.norm(sheep_positions_simulation - r_b, axis=3)

    return np.mean(distances, axis=2)


def calculate_polarization_sim(sheep_velocities_simulation):
    """
    Calculate polarization for sheep flock simulations.

    Parameters:
    -----------
    sheep_velocities_simulation : np.ndarray
        Array of shape (no_it, n_iter, no_shp, 2) containing velocity vectors
        where the last axis contains (vx, vy) components.

    Returns:
    --------
    np.ndarray
        Polarization values of shape (no_it, n_iter)
    """
    vel_norms = np.linalg.norm(sheep_velocities_simulation, axis=3, keepdims=True)
    mean_unit_velocity = np.mean(sheep_velocities_simulation / vel_norms, axis=2)
    return np.linalg.norm(mean_unit_velocity, axis=2)


def calculate_elongation_sim(sheep_positions_simulation, sheep_velocities_simulation):
    """
    Calculate elongation for sheep flock simulations.
    Parameters:
    -----------
    sheep_positions_simulation : np.ndarray
        Shape: (no_it, n_iter, no_shp, 2)
    sheep_velocities_simulation : np.ndarray
        Shape: (no_it, n_iter, no_shp, 2)

    Returns:
    --------
    np.ndarray
        Elongation values of shape (no_it, n_iter)
    """
    # Compute barycenter position and velocity
    barycenter = np.mean(sheep_positions_simulation, axis=2, keepdims=True)
    barycenter_vel = np.mean(sheep_velocities_simulation, axis=2, keepdims=True)

    pos_rel = sheep_positions_simulation - barycenter

    # Normalize barycenter velocity to get direction: v̂_B
    vel_norms = np.linalg.norm(barycenter_vel, axis=3, keepdims=True)
    vel_norms = np.where(vel_norms < 1e-10, 1.0, vel_norms)

    v_hat = barycenter_vel / vel_norms

    # Perpendicular direction: v̂_B⊥ (rotate 90° counterclockwise)
    v_perp = np.stack([-v_hat[..., 1], v_hat[..., 0]], axis=-1)

    # Project positions onto parallel and perpendicular axes
    # ȳ_i = (r_i - r_B) · v̂_B
    y_proj = np.sum(pos_rel * v_hat, axis=3)

    # x̄_i = (r_i - r_B) · v̂_B⊥
    x_proj = np.sum(pos_rel * v_perp, axis=3)

    # Compute length and width
    length = np.ptp(y_proj, axis=2)  # max - min along motion
    width = np.ptp(x_proj, axis=2)  # max - min perpendicular

    # Elongation = LENGTH / WIDTH
    return np.where(width < 1e-10, np.nan, length / width)


def calculate_cohesion(sheep_positions):
    """Calculate cohesion as the mean distance from each sheep to barycenter.

    From the paper (page 8):
    C(t) = (1/N) * sum(||r_i(t) - r_B(t)||)
    where r_B is the barycenter and N is number of sheep.

    Args:
        sheep_positions: (num_sheep, 2) array

    Returns:
        float - cohesion value (mean radius of group)
    """
    barycenter = np.mean(sheep_positions, axis=0)
    distances = np.linalg.norm(sheep_positions - barycenter, axis=1)
    return np.mean(distances)


def calculate_polarization(sheep_velocities):
    """Calculate polarization as average alignment.

    P(t) = (1/N) * ||sum(v_i(t)/||v_i(t)||)||

    Args:
        sheep_velocities: (num_sheep, 2) array

    Returns:
        float - polarization value (0 to 1, where 1 is fully aligned)
    """
    if np.allclose(sheep_velocities, 0):
        return 0.0

    normalized_vels = sheep_velocities / (
            np.linalg.norm(sheep_velocities, axis=1, keepdims=True) + 1e-8)
    avg_direction = np.mean(normalized_vels, axis=0)
    return np.linalg.norm(avg_direction)


def calculate_elongation(sheep_positions, sheep_velocities, barycenter=None):
    """Calculate elongation as a length/width ratio.

    Define a coordinate system in reference frame of barycenter motion:
      ŷ-axis: direction of flock motion (barycenter velocity)
      x̄-axis: perpendicular to motion

    In this coordinate system:
      ȳ_i = d_Bi cos(ψ_Bi) = (r_i - r_B) · v̂_B  (projection along motion)
      x̄_i = d_Bi sin(ψ_Bi) = (r_i - r_B) · v̂_B⊥ (projection perpendicular)

    Then:
      LENGTH = max(ȳ_i) - min(ȳ_i)
      WIDTH = max(x̄_i) - min(x̄_i)
      E(t) = LENGTH / WIDTH

    Args:
        sheep_positions: (num_sheep, 2) array of positions
        sheep_velocities: (num_sheep, 2) array of velocities
        barycenter: (2,) array, calculated if None

    Returns:
        float - elongation value (E > 1 means flock is stretched along motion)

    Raises:
        ValueError: if velocities are zero/invalid or width is zero
    """
    if barycenter is None:
        barycenter = np.mean(sheep_positions, axis=0)

    barycenter_velocity = np.mean(sheep_velocities, axis=0)

    velocity_norm = np.linalg.norm(barycenter_velocity)

    # Unit vector in direction of motion
    direction = barycenter_velocity / velocity_norm

    # Relative positions: r_i - r_B
    relative_pos = sheep_positions - barycenter

    # Project along direction of motion (ŷ-axis)
    proj_along = np.dot(relative_pos, direction)
    length = np.max(proj_along) - np.min(proj_along)

    # Project perpendicular to motion (x̄-axis)
    perp_direction = np.array([-direction[1], direction[0]])
    proj_perp = np.dot(relative_pos, perp_direction)
    width = np.max(proj_perp) - np.min(proj_perp)

    elongation = length / width
    return elongation


def analyze_simulation_metrics(sheep_pos_log, sheep_vel_log):
    """Calculate all metrics over the entire simulation.

    Args:
        sheep_pos_log: (num_frames, num_sheep, 2)
        sheep_vel_log: (num_frames, num_sheep, 2)

    Returns:
        dict with time series for each metric
    """
    num_frames = sheep_pos_log.shape[0]

    cohesion_log = np.zeros(num_frames)
    polarization_log = np.zeros(num_frames)
    elongation_log = np.zeros(num_frames)

    for t in range(num_frames):
        cohesion_log[t] = calculate_cohesion(sheep_pos_log[t])
        polarization_log[t] = calculate_polarization(sheep_vel_log[t])

        elongation_log[t] = calculate_elongation(
            sheep_pos_log[t],
            sheep_vel_log[t]
        )

    return {
        "cohesion": cohesion_log,
        "polarization": polarization_log,
        "elongation": elongation_log,
        "time": np.arange(num_frames),
    }
