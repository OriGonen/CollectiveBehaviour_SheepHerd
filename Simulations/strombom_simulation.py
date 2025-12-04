from Movement_Algorithms.Stormbom_movement_functions import simulate_model_strombom_main

from animation import HerdingAnimation

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# TODO:
# - Add animation with barycenter arrows + individuals
# - Add animation with "trail"
# - Test UI
# - Add recording
# - Refactor all
# NOTE FOR ALL: download ffmpeg for animation!
# https://www.gyan.dev/ffmpeg/builds/

def animate_herding(
        sheep_positions_log,
        dog_positions_log,
        interval=50,  # ms between frames
        box_length=100  # optional, for setting plot boundaries
):
    num_frames = sheep_positions_log.shape[0]
    num_sheep = sheep_positions_log.shape[1]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-box_length, box_length)
    ax.set_ylim(-box_length, box_length)
    ax.set_aspect('equal')
    ax.set_title("Herding Simulation")

    # Scatter plot objects
    sheep_scatter = ax.scatter([], [], s=30, color="blue", label="Sheep")
    dog_scatter = ax.scatter([], [], s=80, color="red", label="Dog")
    ax.legend()

    def update(frame):
        sheep_positions = sheep_positions_log[frame]
        dog_position = dog_positions_log[frame]

        sheep_scatter.set_offsets(sheep_positions)
        dog_scatter.set_offsets(dog_position)

        ax.set_title(f"Herding Simulation — Frame {frame}")
        return sheep_scatter, dog_scatter

    ani = FuncAnimation(fig, update, frames=num_frames,
                        interval=interval, blit=True)
    plt.show()
    print("saving...")
    ani.save("herding_simulation.mp4", fps=30)
    print("saved!")
    return ani

if __name__ == "__main__":
    params = dict(

        num_sheep=30,


        box_length=100.0,

        # interactions distances
        ra_dist=2.0,  # sheep–sheep repulsion distance
        rs_range=25.0,  # dog detection range

        n_neighbors=4,

        # movement per timestep
        d_step=1.0,
        ds=2.0,  # dog is faster than sheep


        h_weight=0.5,
        c_weight=1,
        ra_weight=2.0,
        rs_weight=1.5,

        # noise and grazing
        e_noise=0.03,
        p_move=0.05,


        collecting_offset=0.0,
        driving_offset=10.0,

        # goal
        goal=(0.0, 0.0),

        num_iterations=8000
    )

    sheep_positions_log, dog_positions_log, \
        sheep_velocities_log, dog_velocities_log, \
        dog_speeds_log, collecting_flags, driving_flags, slowing_flags = simulate_model_strombom_main(**params)

    anim = HerdingAnimation(sheep_positions_log, dog_positions_log, sheep_velocities_log, dog_velocities_log, dog_speeds_log=dog_speeds_log)
    anim.run()
    #start_time = time.time()
    #animate_herding(sheep_positions_log, dog_positions_log, interval=40, box_length=box_length)
    #elapsed_time = time.time() - start_time
    #print(f"Elapsed time: {elapsed_time:.4f} seconds")