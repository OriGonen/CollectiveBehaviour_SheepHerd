import pygame
import numpy as np
from collections import deque


class HerdingAnimation:
    def __init__(self, sheep_pos_log, dog_pos_log, sheep_vel_log, dog_vel_log,
                 dog_speeds_log=None, window_size=800, trail_duration=50):
        """
        sheep_vel_log: (num_frames, num_sheep, 2)
        dog_vel_log: (num_frames, 2)
        """
        self.sheep_pos_log = sheep_pos_log
        self.dog_pos_log = dog_pos_log
        self.sheep_vel_log = sheep_vel_log
        self.dog_vel_log = dog_vel_log
        self.dog_speeds_log = dog_speeds_log
        self.window_size = window_size
        self.num_frames = sheep_pos_log.shape[0]
        self.num_sheep = sheep_pos_log.shape[1]
        self.trail_duration = trail_duration

        # State
        self.frame = 0
        self.paused = False
        self.speed = 1.0
        self.render_mode = 0
        self.modes = ["Dots", "Dots + Arrows"]

        # Trail history
        self.sheep_trails = [deque(maxlen=trail_duration) for _ in range(self.num_sheep)]
        self.dog_trail = deque(maxlen=trail_duration)

        # Colors
        self.sheep_color = (0, 0, 255)
        self.dog_color = (255, 0, 0)
        self.sheep_trail_color = (100, 100, 200)
        self.dog_trail_color = (200, 100, 100)
        self.bg_color = (240, 240, 240)
        self.text_color = (0, 0, 0)

    def reset_trails(self):
        """Clear all trail history"""
        for trail in self.sheep_trails:
            trail.clear()
        self.dog_trail.clear()

    def update_trails(self):
        """Add current frame positions to trail history"""
        sheep_pos = self.sheep_pos_log[self.frame]
        dog_pos = self.dog_pos_log[self.frame]

        for i, pos in enumerate(sheep_pos):
            self.sheep_trails[i].append((pos.copy(), self.frame))
        self.dog_trail.append((dog_pos.copy(), self.frame))

    def cycle_render_mode(self):
        """Switch to next render mode"""
        self.render_mode = (self.render_mode + 1) % len(self.modes)
        self.reset_trails()

    def world_to_screen(self, pos, vp_bounds):
        """Convert world coordinates to screen coordinates"""
        vp_x_min, vp_x_max, vp_y_min, vp_y_max = vp_bounds
        x_range = vp_x_max - vp_x_min
        y_range = vp_y_max - vp_y_min

        sx = (pos[0] - vp_x_min) / x_range * self.window_size
        sy = (pos[1] - vp_y_min) / y_range * self.window_size
        return (int(sx), int(sy))

    def render_dots(self, screen, sheep_pos, dog_pos, vp_bounds):
        """Mode 0: Simple circles"""
        for pos in sheep_pos:
            screen_pos = self.world_to_screen(pos, vp_bounds)
            pygame.draw.circle(screen, self.sheep_color, screen_pos, 5)

        dog_screen_pos = self.world_to_screen(dog_pos, vp_bounds)
        pygame.draw.circle(screen, self.dog_color, dog_screen_pos, 8)

    def render_dots_arrows(self, screen, sheep_pos, sheep_vel, dog_pos, dog_vel, vp_bounds):
        """Mode 1: Circles with velocity arrows"""
        arrow_length = 15

        # Sheep
        for pos, vel in zip(sheep_pos, sheep_vel):
            screen_pos = self.world_to_screen(pos, vp_bounds)
            pygame.draw.circle(screen, self.sheep_color, screen_pos, 5)

            # Normalize velocity and scale for display
            vel_norm = np.linalg.norm(vel)
            if vel_norm > 1e-8:
                vel_unit = vel / vel_norm
                arrow_end = np.array(screen_pos) + arrow_length * vel_unit
                pygame.draw.line(screen, self.sheep_color, screen_pos,
                                 tuple(arrow_end.astype(int)), 2)

        # Dog
        dog_screen_pos = self.world_to_screen(dog_pos, vp_bounds)
        pygame.draw.circle(screen, self.dog_color, dog_screen_pos, 8)

        dog_vel_norm = np.linalg.norm(dog_vel)
        if dog_vel_norm > 1e-8:
            dog_vel_unit = dog_vel / dog_vel_norm
            arrow_end = np.array(dog_screen_pos) + arrow_length * dog_vel_unit
            pygame.draw.line(screen, self.dog_color, dog_screen_pos,
                             tuple(arrow_end.astype(int)), 2)

    def render_trails(self, screen, sheep_pos, dog_pos, vp_bounds):
        """Mode 2: Circles with fading trails"""

        def draw_trail(trail, base_color):
            if len(trail) < 2:
                return

            trail_list = list(trail)
            for i in range(len(trail_list) - 1):
                pos1, _ = trail_list[i]
                pos2, _ = trail_list[i + 1]

                # Alpha fade: oldest → transparent, newest → opaque
                alpha = int(255 * (i + 1) / len(trail_list))

                screen_pos1 = self.world_to_screen(pos1, vp_bounds)
                screen_pos2 = self.world_to_screen(pos2, vp_bounds)

                # Create surface with alpha
                line_surf = pygame.Surface((self.window_size, self.window_size),
                                           pygame.SRCALPHA)
                color_with_alpha = (*base_color, alpha)
                pygame.draw.line(line_surf, color_with_alpha, screen_pos1,
                                 screen_pos2, 2)
                screen.blit(line_surf, (0, 0))

        # Draw each sheep trail
        for sheep_trail in self.sheep_trails:
            draw_trail(sheep_trail, self.sheep_trail_color)

        # Draw dog trail
        draw_trail(self.dog_trail, self.dog_trail_color)

        # Draw current positions on top
        for pos in sheep_pos:
            screen_pos = self.world_to_screen(pos, vp_bounds)
            pygame.draw.circle(screen, self.sheep_color, screen_pos, 5)

        dog_screen_pos = self.world_to_screen(dog_pos, vp_bounds)
        pygame.draw.circle(screen, self.dog_color, dog_screen_pos, 8)

    def compute_viewport(self, sheep_pos, dog_pos, padding=30):
        """Calculate viewport bounds centered on flock"""
        flock_center = np.mean(sheep_pos, axis=0)
        all_pos = np.vstack([sheep_pos, dog_pos.reshape(1, -1)])

        x_min, x_max = all_pos[:, 0].min(), all_pos[:, 0].max()
        y_min, y_max = all_pos[:, 1].min(), all_pos[:, 1].max()

        x_range = max(x_max - x_min + padding, padding * 2)
        y_range = max(y_max - y_min + padding, padding * 2)

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        return (x_center - x_range / 2, x_center + x_range / 2,
                y_center - y_range / 2, y_center + y_range / 2), flock_center

    def run(self):
        """Main animation loop"""
        pygame.init()
        screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Herding Simulation")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 22)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    if event.key == pygame.K_EQUALS or event.key == pygame.K_UP:
                        self.speed = min(self.speed + 0.5, 5.0)
                    if event.key == pygame.K_MINUS or event.key == pygame.K_DOWN:
                        self.speed = max(self.speed - 0.5, 0.5)
                    if event.key == pygame.K_r:
                        self.frame = 0
                        self.reset_trails()
                    if event.key == pygame.K_m:
                        self.cycle_render_mode()
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False

            if not self.paused:
                self.frame = (self.frame + int(self.speed)) % self.num_frames

            # Update trails
            self.update_trails()

            # Data extraction
            sheep_pos = self.sheep_pos_log[self.frame]
            sheep_vel = self.sheep_vel_log[self.frame]
            dog_pos = self.dog_pos_log[self.frame]
            dog_vel = self.dog_vel_log[self.frame]

            vp_bounds, flock_center = self.compute_viewport(sheep_pos, dog_pos)

            # Render
            screen.fill(self.bg_color)

            if self.render_mode == 0:
                self.render_dots(screen, sheep_pos, dog_pos, vp_bounds)
            elif self.render_mode == 1:
                self.render_dots_arrows(screen, sheep_pos, sheep_vel, dog_pos, dog_vel, vp_bounds)
            #elif self.render_mode == 2:
            #    self.render_trails(screen, sheep_pos, dog_pos, vp_bounds)

            # HUD
            status = "PAUSED" if self.paused else "RUNNING"
            dog_speed_str = f"  Dog: {self.dog_speeds_log[self.frame]:.2f}" if self.dog_speeds_log is not None else ""

            hud_texts = [
                f"Frame: {self.frame}/{self.num_frames}  Speed: {self.speed:.1f}x  {status}  Mode: {self.modes[self.render_mode]}",
                f"Flock: ({flock_center[0]:.1f}, {flock_center[1]:.1f}){dog_speed_str}",
                "SPACE: pause | +/-: speed | R: reset | M: mode | Q: quit"
            ]

            for i, text in enumerate(hud_texts):
                surf = font.render(text, True, self.text_color)
                screen.blit(surf, (10, 10 + i * 25))

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
