import numpy as np
import time
import pygame

from movement_functions import simulate_model

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# TODO:
# - Add animation with direction arrows for individuals
# - Add animation with barycenter arrows + individuals
# - Add animation with "trail"

def animate_pygame(sheep_pos_log, dog_pos_log, dog_speeds_log=None,
                   window_size=800, sheep_color=(0, 0, 255),
                   dog_color=(255, 0, 0), padding=30):
    """
    Interactive pygame animation.
    Controls:
    - SPACE: pause/resume
    - +/-: speed up/slow down
    - R: reset to frame 0
    - Q/ESC: quit
    """
    pygame.init()
    screen = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("Herding Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    num_frames = sheep_pos_log.shape[0]
    frame = 0
    paused = False
    speed = 1.0  # playback speed multiplier

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_EQUALS or event.key == pygame.K_UP:
                    speed = min(speed + 0.5, 5.0)
                if event.key == pygame.K_MINUS or event.key == pygame.K_DOWN:
                    speed = max(speed - 0.5, 0.5)
                if event.key == pygame.K_r:
                    frame = 0
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

        if not paused:
            frame = (frame + int(speed)) % num_frames

        # Extract data
        sheep_pos = sheep_pos_log[frame]
        dog_pos = dog_pos_log[frame]

        # Calculate viewport (centered on flock centroid)
        flock_center = np.mean(sheep_pos, axis=0)
        all_pos = np.vstack([sheep_pos, dog_pos.reshape(1, -1)])
        x_min, x_max = all_pos[:, 0].min(), all_pos[:, 0].max()
        y_min, y_max = all_pos[:, 1].min(), all_pos[:, 1].max()

        x_range = max(x_max - x_min + padding, padding * 2)
        y_range = max(y_max - y_min + padding, padding * 2)

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        # Viewport bounds
        vp_x_min = x_center - x_range / 2
        vp_x_max = x_center + x_range / 2
        vp_y_min = y_center - y_range / 2
        vp_y_max = y_center + y_range / 2

        # Conversion function: world coords to screen coords
        def world_to_screen(pos):
            sx = (pos[0] - vp_x_min) / x_range * window_size
            sy = (pos[1] - vp_y_min) / y_range * window_size
            return (int(sx), int(sy))

        # Draw
        screen.fill((240, 240, 240))

        # Draw sheep
        for pos in sheep_pos:
            screen_pos = world_to_screen(pos)
            pygame.draw.circle(screen, sheep_color, screen_pos, 5)

        # Draw dog
        dog_screen_pos = world_to_screen(dog_pos)
        pygame.draw.circle(screen, dog_color, dog_screen_pos, 8)

        # Draw flock centroid (small cross)
        center_screen = world_to_screen(flock_center)
        pygame.draw.line(screen, (100, 100, 100),
                         (center_screen[0] - 5, center_screen[1]),
                         (center_screen[0] + 5, center_screen[1]), 1)
        pygame.draw.line(screen, (100, 100, 100),
                         (center_screen[0], center_screen[1] - 5),
                         (center_screen[0], center_screen[1] + 5), 1)

        # HUD text
        status = "PAUSED" if paused else "RUNNING"
        dog_speed_str = f"Dog speed: {dog_speeds_log[frame]:.2f}" if dog_speeds_log is not None else ""

        texts = [
            f"Frame: {frame}/{num_frames}  Speed: {speed:.1f}x  {status}",
            f"Flock center: ({flock_center[0]:.1f}, {flock_center[1]:.1f})",
            dog_speed_str,
            "SPACE: pause | +/-: speed | R: reset | Q: quit"
        ]

        for i, text in enumerate(texts):
            if text:
                surf = font.render(text, True, (0, 0, 0))
                screen.blit(surf, (10, 10 + i * 25))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


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
    # Example parameters — replace with your real ones
    num_sheep = 10
    box_length = 50

    sheep_positions_log, dog_positions_log, \
        sheep_velocities_log, dog_velocities_log, \
        dog_speeds_log, collecting_flags, driving_flags, slowing_flags = simulate_model(
        num_sheep,
        box_length=box_length,
        sheep_repulsion_radius=1.0,
        dog_repulsion_radius=5.0,
        num_neighbors_for_attraction=10,
        num_random_attraction_neighbors=5,
        num_alignment_neighbors=5,
        sheep_speed=0.1,
        dog_speed=0.25,
        persistence_weight=0.5,
        sheep_repulsion_weight=1.5,
        dog_repulsion_weight=2.0,
        noise_weight=0.05,
        attraction_weight=2.0,
        alignment_weight=0.1,
        non_cohesive_distance=3.0,
        driving_offset=2.0,
        collecting_offset=1.5,
        num_iterations=800
    )

    animate_pygame(sheep_positions_log, dog_positions_log, dog_speeds_log=dog_speeds_log)

    start_time = time.time()
    #animate_herding(sheep_positions_log, dog_positions_log, interval=40, box_length=box_length)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    # TODO:
    # - go over the code to make sure it's fine. comment: the dog sometimes is hesitating between two stray sheep but I think it makes sense with its movement algorithm
    # - make sure the agents are in the box always (maybe make it cyclic? idk)
    # NOTE FOR ALL: download ffmpeg for animation!
    # https://www.gyan.dev/ffmpeg/builds/