import pygame
import numpy as np
import csv

from utils.metrics import calculate_cohesion, calculate_polarization

FONT_SIZE = 30

# TODO: Decide if we want scaling or just better calculation of the game window

class HerdingAnimation:
    def __init__(self, sheep_pos_log, dog_pos_log, sheep_vel_log, dog_vel_log,
                 dog_speeds_log=None, window_size=1200, show_metrics=True):
        self.sheep_pos_log = sheep_pos_log
        self.dog_pos_log = dog_pos_log
        self.sheep_vel_log = sheep_vel_log
        self.dog_vel_log = dog_vel_log
        self.dog_speeds_log = dog_speeds_log
        self.window_size = window_size
        self.num_frames = sheep_pos_log.shape[0]
        self.num_sheep = sheep_pos_log.shape[1]
        self.show_metrics = show_metrics

        # Layout parameters
        self.overlay_width = 200
        self.margin = 10
        self.right_margin = 20
        self.left_margin = 20
        self.game_area_x = self.overlay_width + self.margin
        self.game_area_width = self.window_size - self.game_area_x - self.right_margin
        self.game_area_y = self.left_margin
        self.game_area_height = self.window_size - self.left_margin - self.game_area_y

        # Calculate global bounds once
        self.global_bounds, self.global_center = self._compute_global_bounds()

        # Calculate game viewport (fit bounded box into game area)
        self._calculate_game_viewport()

        if self.show_metrics:
            self.cohesion_log = np.zeros(self.num_frames)
            self.polarization_log = np.zeros(self.num_frames)
            for t in range(self.num_frames):
                self.cohesion_log[t] = calculate_cohesion(sheep_pos_log[t])
                self.polarization_log[t] = calculate_polarization(sheep_vel_log[t])

        # State - discrete frame handling
        self.frame = 0
        self.frame_accumulator = 0.0
        self.paused = False
        self.speed = 1.0
        self.render_mode = 0
        self.modes = ["Dots", "Dots + Arrows"]

        # Colors
        self.sheep_color = (0, 0, 255)
        self.dog_color = (255, 0, 0)
        self.bg_color = (240, 240, 240)
        self.overlay_bg_color = (220, 220, 220)
        self.text_color = (0, 0, 0)
        self.bounds_color = (100, 100, 100)

    def _compute_global_bounds(self, padding=50):
        """Calculate fixed bounds for the entire simulation"""
        all_positions = np.vstack([
            self.sheep_pos_log.reshape(-1, 2),
            self.dog_pos_log.reshape(-1, 2)
        ])

        x_min, y_min = all_positions.min(axis=0) - padding
        x_max, y_max = all_positions.max(axis=0) + padding

        return (x_min, x_max, y_min, y_max), all_positions.mean(axis=0)

    def _calculate_game_viewport(self):
        """Calculate a viewport that fits bounded box into the game area"""
        vp_x_min, vp_x_max, vp_y_min, vp_y_max = self.global_bounds

        bounds_width = vp_x_max - vp_x_min
        bounds_height = vp_y_max - vp_y_min

        bounds_aspect = bounds_width / bounds_height
        game_aspect = self.game_area_width / self.game_area_height

        if bounds_aspect > game_aspect:
            # Bounds wider: fit to width
            game_width = self.game_area_width
            game_height = game_width / bounds_aspect
        else:
            # Bounds taller: fit to height
            game_height = self.game_area_height
            game_width = game_height * bounds_aspect

        # Center in the game area
        self.game_offset_x = self.game_area_x + (self.game_area_width - game_width) / 2
        self.game_offset_y = self.game_area_y + (self.game_area_height - game_height) / 2
        self.game_scale_x = game_width / bounds_width
        self.game_scale_y = game_height / bounds_height

    def world_to_screen(self, pos):
        """Convert world coordinates to screen coordinates in the game area"""
        vp_x_min, vp_x_max, vp_y_min, vp_y_max = self.global_bounds

        sx = self.game_offset_x + (pos[0] - vp_x_min) * self.game_scale_x
        sy = self.game_offset_y + (pos[1] - vp_y_min) * self.game_scale_y

        return int(sx), int(sy)

    def update_frame(self):
        """Update frame with accumulator-based speed handling"""
        if self.paused:
            return

        # Accumulate speed
        self.frame_accumulator += self.speed

        # Advance frame(s) when accumulator >= 1.0
        while self.frame_accumulator >= 1.0:
            self.frame_accumulator -= 1.0
            self.frame = (self.frame + 1) % self.num_frames

    def get_current_frame_data(self):
        """Get data for current frame"""
        return (
            self.sheep_pos_log[self.frame],
            self.sheep_vel_log[self.frame],
            self.dog_pos_log[self.frame],
            self.dog_vel_log[self.frame]
        )

    def render_bounds(self, screen):
        """Render simulation bounds in the game area"""
        vp_x_min, vp_x_max, vp_y_min, vp_y_max = self.global_bounds

        corners = [
            self.world_to_screen([vp_x_min, vp_y_min]),
            self.world_to_screen([vp_x_max, vp_y_min]),
            self.world_to_screen([vp_x_max, vp_y_max]),
            self.world_to_screen([vp_x_min, vp_y_max])
        ]

        pygame.draw.polygon(screen, self.bounds_color, corners, 2)

    def render_dots(self, screen, sheep_pos, dog_pos):
        """Mode 0: Simple circles"""
        for pos in sheep_pos:
            screen_pos = self.world_to_screen(pos)
            pygame.draw.circle(screen, self.sheep_color, screen_pos, 5)

        dog_screen_pos = self.world_to_screen(dog_pos)
        pygame.draw.circle(screen, self.dog_color, dog_screen_pos, 8)

    def render_dots_arrows(self, screen, sheep_pos, sheep_vel, dog_pos, dog_vel):
        """Mode 1: Circles with velocity arrows"""
        arrow_length = 15

        # Sheep
        for pos, vel in zip(sheep_pos, sheep_vel):
            screen_pos = self.world_to_screen(pos)
            pygame.draw.circle(screen, self.sheep_color, screen_pos, 5)

            vel_norm = np.linalg.norm(vel)
            if vel_norm > 1e-8:
                vel_unit = vel / vel_norm
                arrow_end = np.array(screen_pos) + arrow_length * vel_unit
                pygame.draw.line(screen, self.sheep_color, screen_pos,
                                 tuple(arrow_end.astype(int)), 2)

        # Dog
        dog_screen_pos = self.world_to_screen(dog_pos)
        pygame.draw.circle(screen, self.dog_color, dog_screen_pos, 8)

        dog_vel_norm = np.linalg.norm(dog_vel)
        if dog_vel_norm > 1e-8:
            dog_vel_unit = dog_vel / dog_vel_norm
            arrow_end = np.array(dog_screen_pos) + arrow_length * dog_vel_unit
            pygame.draw.line(screen, self.dog_color, dog_screen_pos,
                             tuple(arrow_end.astype(int)), 2)

    def export_data(self):
        """Export metrics and simulation data to CSV"""
        timestamp = pygame.time.get_ticks()

        # Export metrics
        metrics_file = self.export_dir / f"metrics_{timestamp}.csv"
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'cohesion', 'polarization', 'dog_speed'])

            for frame in range(self.num_frames):
                dog_speed = self.dog_speeds_log[frame] if self.dog_speeds_log is not None else 0
                writer.writerow([
                    frame,
                    self.cohesion_log[frame] if self.show_metrics else 0,
                    self.polarization_log[frame] if self.show_metrics else 0,
                    dog_speed
                ])

        # Export positions
        positions_file = self.export_dir / f"positions_{timestamp}.csv"
        with open(positions_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'entity_type', 'entity_id', 'x', 'y', 'vx', 'vy'])

            for frame in range(self.num_frames):
                # Sheep
                for sheep_id in range(self.num_sheep):
                    writer.writerow([
                        frame, 'sheep', sheep_id,
                        self.sheep_pos_log[frame, sheep_id, 0],
                        self.sheep_pos_log[frame, sheep_id, 1],
                        self.sheep_vel_log[frame, sheep_id, 0],
                        self.sheep_vel_log[frame, sheep_id, 1]
                    ])

                # Dog
                writer.writerow([
                    frame, 'dog', 0,
                    self.dog_pos_log[frame, 0],
                    self.dog_pos_log[frame, 1],
                    self.dog_vel_log[frame, 0],
                    self.dog_vel_log[frame, 1]
                ])

    def render_overlay(self, screen, font):
        """Render left sidebar with information"""
        overlay_surface = pygame.Surface((self.overlay_width, self.window_size))
        overlay_surface.fill(self.overlay_bg_color)

        y_offset = 10
        line_height = FONT_SIZE

        status = "PAUSED" if self.paused else "RUNNING"
        texts = [
            f"Status: {status}",
            f"Frame: {self.frame}/{self.num_frames - 1}",
            f"Speed: {self.speed:.2f}x",
            f"Mode: {self.modes[self.render_mode]}",
            "",
            "CONTROLS:",
            "SPACE: pause",
            "+/-: speed",
            "R: reset",
            "M: mode",
            "",
            "E: export",
            "Q: quit",
        ]

        for text in texts:
            if text == "":
                y_offset += 6
            else:
                surf = font.render(text, True, self.text_color)
                overlay_surface.blit(surf, (10, y_offset))
                y_offset += line_height

        # Add metrics at the bottom of overlay
        if self.show_metrics:
            dog_speed = self.dog_speeds_log[self.frame] if self.dog_speeds_log is not None else 0
            metrics_texts = [
                "",
                f"Cohesion: {self.cohesion_log[self.frame]:.3f}",
                f"Polar: {self.polarization_log[self.frame]:.3f}",
                f"Dog Spd: {dog_speed:.2f}"
            ]

            for text in metrics_texts:
                if text == "":
                    y_offset += 20
                else:
                    surf = font.render(text, True, self.text_color)
                    overlay_surface.blit(surf, (10, y_offset))
                    y_offset += line_height

        screen.blit(overlay_surface, (0, 0))

    def render_game_area(self, screen, sheep_pos, sheep_vel, dog_pos, dog_vel):
        """Render game area with bounds and entities"""
        # Draw background for the game area
        game_rect = pygame.Rect(self.game_area_x, self.game_area_y,
                                self.game_area_width, self.game_area_height)
        pygame.draw.rect(screen, self.bg_color, game_rect)

        # Draw bounds
        self.render_bounds(screen)

        # Render entities
        if self.render_mode == 0:
            self.render_dots(screen, sheep_pos, dog_pos)
        elif self.render_mode == 1:
            self.render_dots_arrows(screen, sheep_pos, sheep_vel, dog_pos, dog_vel)

    def run(self):
        """Main animation loop"""
        pygame.init()
        screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Herding Simulation")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, FONT_SIZE)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    if event.key == pygame.K_EQUALS or event.key == pygame.K_UP:
                        self.speed = min(self.speed + 0.1, 10.0)
                    if event.key == pygame.K_MINUS or event.key == pygame.K_DOWN:
                        self.speed = max(self.speed - 0.1, 0.1)
                    if event.key == pygame.K_r:
                        self.frame = 0
                        self.frame_accumulator = 0.0
                    if event.key == pygame.K_m:
                        self.render_mode = (self.render_mode + 1) % len(self.modes)

                    # Export controls
                    if event.key == pygame.K_e:
                        self.export_data()

                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False

            # Update frame
            self.update_frame()

            # Get current frame data
            sheep_pos, sheep_vel, dog_pos, dog_vel = self.get_current_frame_data()

            # Render
            screen.fill((200, 200, 200))
            self.render_game_area(screen, sheep_pos, sheep_vel, dog_pos, dog_vel)
            self.render_overlay(screen, font)

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()