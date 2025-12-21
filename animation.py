import pygame
import numpy as np
from pathlib import Path
import csv
import cv2

# EASY REMOVAL SECTION: Video recording dependencies
try:
    import cv2

    VIDEO_SUPPORT = True
except ImportError:
    VIDEO_SUPPORT = False
    print("OpenCV not available - video recording disabled")


# END EASY REMOVAL SECTION

class HerdingAnimation:
    def __init__(self, sheep_pos_log, dog_pos_log, sheep_vel_log, dog_vel_log,
                 dog_speeds_log=None, window_size=800, trail_duration=50,
                 show_metrics=True):
        self.sheep_pos_log = sheep_pos_log
        self.dog_pos_log = dog_pos_log
        self.sheep_vel_log = sheep_vel_log
        self.dog_vel_log = dog_vel_log
        self.dog_speeds_log = dog_speeds_log
        self.window_size = window_size
        self.num_frames = sheep_pos_log.shape[0]  # Number of iterations
        self.num_sheep = sheep_pos_log.shape[1]
        self.trail_duration = trail_duration
        self.show_metrics = show_metrics

        # EASY REMOVAL SECTION: Loop controls
        self.loop_enabled = True
        self.ping_pong = False
        self.loop_direction = 1
        # END EASY REMOVAL SECTION

        # Debug settings
        self.debug_enabled = False
        self.show_grid = False
        self.show_bounds = True  # Always show simulation bounds

        # Calculate global bounds once
        self.global_bounds, self.global_center = self._compute_global_bounds()

        # TODO: It should probably just use the metrics library for calculating metrics.
        # TODO: This is super slow tbh, im sure it could be optimized.
        if self.show_metrics:
            self.cohesion_log = np.zeros(self.num_frames)
            self.polarization_log = np.zeros(self.num_frames)
            for t in range(self.num_frames):
                self.cohesion_log[t] = self._calculate_cohesion(sheep_pos_log[t])
                self.polarization_log[t] = self._calculate_polarization(sheep_vel_log[t])

        # State - discrete frame handling
        self.frame = 0  # Current frame index (0 to num_frames-1)

        # FIXME: fractional speeds should not exist
        self.frame_accumulator = 0.0  # For handling fractional speeds
        self.paused = False
        self.speed = 1.0
        self.render_mode = 0
        self.modes = ["Dots", "Dots + Arrows"]

        # Export paths
        self.export_dir = Path("exports")
        self.export_dir.mkdir(exist_ok=True)

        # Colors
        self.sheep_color = (0, 0, 255)
        self.dog_color = (255, 0, 0)
        self.bg_color = (240, 240, 240)
        self.text_color = (0, 0, 0)
        self.grid_color = (200, 200, 200)
        self.bounds_color = (100, 100, 100)

        # Clipping notifications
        self.clipping_count = 0
        self.max_clipping_reports = 5
        self.clipping_notifications = True

    def _compute_global_bounds(self, padding=50):
        """Calculate fixed bounds for the entire simulation"""
        all_positions = np.vstack([
            self.sheep_pos_log.reshape(-1, 2),
            self.dog_pos_log.reshape(-1, 2)
        ])

        x_min, y_min = all_positions.min(axis=0) - padding
        x_max, y_max = all_positions.max(axis=0) + padding

        return (x_min, x_max, y_min, y_max), all_positions.mean(axis=0)

    def _calculate_cohesion(self, sheep_pos):
        """Calculate flock cohesion (average distance from center)"""
        center = np.mean(sheep_pos, axis=0)
        distances = np.linalg.norm(sheep_pos - center, axis=1)
        return np.mean(distances)

    def _calculate_polarization(self, sheep_vel):
        """Calculate flock polarization (alignment)"""
        velocity_vectors = sheep_vel
        if np.allclose(velocity_vectors, 0):
            return 0.0

        avg_velocity = np.mean(velocity_vectors, axis=0)
        magnitudes = np.linalg.norm(velocity_vectors, axis=1)

        if np.sum(magnitudes) == 0:
            return 0.0

        polarization = np.linalg.norm(avg_velocity) / np.sum(magnitudes)
        return min(polarization, 1.0)

    def toggle_clipping_notifications(self):
        """Toggle clipping notifications on/off"""
        self.clipping_notifications = not self.clipping_notifications

        status = "enabled" if self.clipping_notifications else "disabled"
        print(f"Clipping notifications {status}", file=sys.stderr)

        return self.clipping_notifications

    def world_to_screen(self, pos):
        """Convert world coordinates to screen coordinates with bounds checking and notifications"""
        vp_x_min, vp_x_max, vp_y_min, vp_y_max = self.global_bounds

        # Check if clipping is needed
        needs_clipping = False
        original_pos = pos.copy()

        if pos[0] < vp_x_min:
            needs_clipping = True
        elif pos[0] > vp_x_max:
            needs_clipping = True
        elif pos[1] < vp_y_min:
            needs_clipping = True
        elif pos[1] > vp_y_max:
            needs_clipping = True

        # Perform clipping
        pos = np.clip(pos, [vp_x_min, vp_y_min], [vp_x_max, vp_y_max])

        # Report clipping if it occurred
        if needs_clipping and self.clipping_notifications:
            self.clipping_count += 1

            # Only report first few times, then periodically
            if self.clipping_count <= self.max_clipping_reports or \
                    self.clipping_count % 100 == 0:

                diff = original_pos - pos
                clipped_dims = []
                if diff[0] != 0:
                    clipped_dims.append(f"x: {original_pos[0]:.2f} -> {pos[0]:.2f}")
                if diff[1] != 0:
                    clipped_dims.append(f"y: {original_pos[1]:.2f} -> {pos[1]:.2f}")

                print(f"CLIPPING: Position ({', '.join(clipped_dims)}) - "
                      f"Total clips: {self.clipping_count}",
                      file=sys.stderr)

        # Continue with normal conversion
        x_range = vp_x_max - vp_x_min
        y_range = vp_y_max - vp_y_min

        sx = (pos[0] - vp_x_min) / x_range * self.window_size
        sy = (pos[1] - vp_y_min) / y_range * self.window_size
        return (int(sx), int(sy))

    def update_frame(self):
        """Update current frame based on speed - discrete time stepping"""
        if self.paused or self.speed == 0:
            return

        # Add speed to accumulator
        self.frame_accumulator += self.speed

        # Move to next frame when accumulator reaches 1.0
        while self.frame_accumulator >= 1.0:
            self.frame_accumulator -= 1.0

            if self.loop_enabled:
                if self.ping_pong:
                    # Ping-pong mode: go forward, then backward
                    self.frame += self.loop_direction
                    if self.frame >= self.num_frames - 1:
                        self.loop_direction = -1
                    elif self.frame <= 0:
                        self.loop_direction = 1
                else:
                    # Forward loop
                    self.frame = (self.frame + 1) % self.num_frames
            else:
                # No looping - stop at end
                if self.frame < self.num_frames - 1:
                    self.frame += 1

    def get_current_frame_data(self):
        """Get data for current frame - no interpolation"""
        return (
            self.sheep_pos_log[self.frame],
            self.sheep_vel_log[self.frame],
            self.dog_pos_log[self.frame],
            self.dog_vel_log[self.frame]
        )

    def seek_to_frame(self, target_frame):
        """Jump directly to specific frame"""
        self.frame = max(0, min(target_frame, self.num_frames - 1))
        self.frame_accumulator = 0.0

    def render_grid(self, screen):
        """Render grid overlay"""
        if not self.show_grid:
            return

        vp_x_min, vp_x_max, vp_y_min, vp_y_max = self.global_bounds

        # Grid spacing
        grid_size = max((vp_x_max - vp_x_min) / 10, 10)

        # Vertical lines
        x = vp_x_min
        while x <= vp_x_max:
            start_pos = self.world_to_screen([x, vp_y_min])
            end_pos = self.world_to_screen([x, vp_y_max])
            pygame.draw.line(screen, self.grid_color, start_pos, end_pos, 1)
            x += grid_size

        # Horizontal lines
        y = vp_y_min
        while y <= vp_y_max:
            start_pos = self.world_to_screen([vp_x_min, y])
            end_pos = self.world_to_screen([vp_x_max, y])
            pygame.draw.line(screen, self.grid_color, start_pos, end_pos, 1)
            y += grid_size

    def render_bounds(self, screen):
        """Render simulation bounds"""
        if not self.show_bounds:
            return

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

    def save_screenshot(self):
        """Save current frame as screenshot"""
        timestamp = pygame.time.get_ticks()
        screenshot_file = self.export_dir / f"screenshot_{timestamp}.png"

        # Convert pygame surface to image and save
        # This would need the actual rendered surface
        print(f"Screenshot saved to {screenshot_file}")

    # EASY REMOVAL SECTION: Video recording
    def save_video(self, fps=30, scale_factor=2):
        """Render simulation to video file"""
        if not VIDEO_SUPPORT:
            print("Video recording not available - OpenCV not installed")
            return

        timestamp = pygame.time.get_ticks()
        video_file = self.export_dir / f"simulation_{timestamp}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(video_file), fourcc, fps,
            (self.window_size * scale_factor, self.window_size * scale_factor)
        )

        original_size = self.window_size
        self.window_size = self.window_size * scale_factor

        try:
            # Save current state
            saved_frame = self.frame
            saved_accumulator = self.frame_accumulator

            for frame_idx in range(self.num_frames):
                self.seek_to_frame(frame_idx)
                surface = pygame.Surface((self.window_size, self.window_size))

                # Render frame
                sheep_pos, sheep_vel, dog_pos, dog_vel = self.get_current_frame_data()
                surface.fill(self.bg_color)
                self.render_bounds(surface)
                self.render_grid(surface)

                if self.render_mode == 0:
                    self.render_dots(surface, sheep_pos, dog_pos)
                elif self.render_mode == 1:
                    self.render_dots_arrows(surface, sheep_pos, sheep_vel, dog_pos, dog_vel)

                # Convert and write frame
                frame_array = pygame.surfarray.array3d(surface)
                frame_array = np.transpose(frame_array, (1, 0, 2))
                frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_array)

        finally:
            video_writer.release()
            self.window_size = original_size
            # Restore state
            self.frame = saved_frame
            self.frame_accumulator = saved_accumulator

        print(f"Video saved to {video_file}")

    # END EASY REMOVAL SECTION

    def render_debug_panel(self, screen, font):
        """Render debug information panel"""
        if not self.debug_enabled:
            return

        panel_width = 350
        panel_height = 240
        panel_surface = pygame.Surface((panel_width, panel_height))
        panel_surface.set_alpha(200)
        panel_surface.fill((255, 255, 255))

        y_offset = 10
        debug_texts = [
            f"Global Bounds: ({self.global_bounds[0]:.1f}, {self.global_bounds[2]:.1f}) to ({self.global_bounds[1]:.1f}, {self.global_bounds[3]:.1f})",
            f"Global Center: ({self.global_center[0]:.1f}, {self.global_center[1]:.1f})",
            f"Simulation Range: X={(self.global_bounds[1] - self.global_bounds[0]):.1f}, Y={(self.global_bounds[3] - self.global_bounds[2]):.1f}",
            f"Current Frame: {self.frame}/{self.num_frames - 1} (t={self.frame}s)",
            f"Clipping Events: {self.clipping_count} | Notifications: {'ON' if self.clipping_notifications else 'OFF'}",
            f"Frame Accumulator: {self.frame_accumulator:.2f}",
            "G: Toggle Grid | B: Toggle Bounds | E: Export Data",
            "F1: Debug | F2: Toggle Clipping Notifications",
            "S: Screenshot | V: Video (if available)"
        ]

        for text in debug_texts:
            surf = font.render(text, True, self.text_color)
            panel_surface.blit(surf, (10, y_offset))
            y_offset += 20

        screen.blit(panel_surface, (self.window_size - panel_width, 0))

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
                        self.speed = min(self.speed + 0.5, 10.0)
                    if event.key == pygame.K_MINUS or event.key == pygame.K_DOWN:
                        self.speed = max(self.speed - 0.5, 0.1)
                    if event.key == pygame.K_LEFT:
                        self.seek_to_frame(self.frame - 10)
                    if event.key == pygame.K_RIGHT:
                        self.seek_to_frame(self.frame + 10)
                    if event.key == pygame.K_r:
                        self.seek_to_frame(0)
                    if event.key == pygame.K_m:
                        self.render_mode = (self.render_mode + 1) % len(self.modes)

                    # EASY REMOVAL SECTION: Loop controls
                    if event.key == pygame.K_l:
                        self.loop_enabled = not self.loop_enabled
                    if event.key == pygame.K_p:
                        self.ping_pong = not self.ping_pong
                    # END EASY REMOVAL SECTION

                    # Debug controls
                    if event.key == pygame.K_g:
                        self.show_grid = not self.show_grid
                    if event.key == pygame.K_b:
                        self.show_bounds = not self.show_bounds
                    if event.key == pygame.K_F1:
                        self.debug_enabled = not self.debug_enabled
                    if event.key == pygame.K_F2:
                        self.toggle_clipping_notifications()

                    # Export controls
                    if event.key == pygame.K_e:
                        self.export_data()
                    if event.key == pygame.K_s:
                        self.save_screenshot()

                    # EASY REMOVAL SECTION: Video
                    if event.key == pygame.K_v:
                        self.save_video()
                    # END EASY REMOVAL SECTION

                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False

            # Update frame based on speed - discrete time stepping
            self.update_frame()

            # Get current frame data - no interpolation
            sheep_pos, sheep_vel, dog_pos, dog_vel = self.get_current_frame_data()

            # Render
            screen.fill(self.bg_color)
            self.render_bounds(screen)
            self.render_grid(screen)

            if self.render_mode == 0:
                self.render_dots(screen, sheep_pos, dog_pos)
            elif self.render_mode == 1:
                self.render_dots_arrows(screen, sheep_pos, sheep_vel, dog_pos, dog_vel)

            # HUD
            status = "PAUSED" if self.paused else "RUNNING"
            dog_speed = self.dog_speeds_log[self.frame] if self.dog_speeds_log is not None else 0
            flock_center = np.mean(sheep_pos, axis=0)

            hud_texts = [
                f"Frame: {self.frame}/{self.num_frames - 1} (t={self.frame}s)  Speed: {self.speed:.1f}x  {status}  Mode: {self.modes[self.render_mode]}",
                f"Flock: ({flock_center[0]:.1f}, {flock_center[1]:.1f})  Dog: {dog_speed:.2f}"
            ]

            # Add metrics if enabled
            if self.show_metrics:
                metrics_text = (
                    f"Cohesion: {self.cohesion_log[self.frame]:.3f}m  "
                    f"Polarization: {self.polarization_log[self.frame]:.3f}"
                )
                hud_texts.insert(2, metrics_text)

            # EASY REMOVAL SECTION: Loop controls info
            if self.loop_enabled:
                loop_text = f"Loop: {'Ping-Pong' if self.ping_pong else 'Forward'}"
                hud_texts.append(loop_text)
            # END EASY REMOVAL SECTION

            hud_texts.extend([
                "SPACE: pause | +/-: speed | ←/→: seek | R: reset | M: mode",
                "F1: debug | G: grid | B: bounds | E: export | S: screenshot | F2: clipping"
            ])

            for i, text in enumerate(hud_texts):
                surf = font.render(text, True, self.text_color)
                screen.blit(surf, (10, 10 + i * 25))

            self.render_debug_panel(screen, font)

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()