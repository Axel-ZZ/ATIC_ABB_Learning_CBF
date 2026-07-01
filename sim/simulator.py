from __future__ import annotations
import pygame
import numpy as np
import math
import sys
from typing import List, Tuple

# =====================================================================
# 2. ENVIRONMENT AND UTILS IMPORTS
# =====================================================================
try:
    from environments.environment import Environment, build_env
except ImportError:
    # Safe fallback if run outside of repo directory tree
    from dataclasses import dataclass
    print("sdfsdafgdsfsdf") 
    @dataclass
    class Environment:
        name: str
        bounds: Tuple[Tuple[float, float], Tuple[float, float]]
        obstacles: List
        start: np.ndarray
        goal: np.ndarray
        robot_radius: float = 0.1

    # Simple inline mock of build_env for local testing robustness
    class LocalWallObstacle:
        def __init__(self, xmin, xmax, ymin, ymax):
            self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        def collides(self, pt, r):
            cx = max(self.xmin, min(pt[0], self.xmax))
            cy = max(self.ymin, min(pt[1], self.ymax))
            return math.hypot(pt[0] - cx, pt[1] - cy) < r

    def build_env(name: str) -> Environment:
        walls = [
            LocalWallObstacle(0.5, 0.8, 2.9, 6.0),
            LocalWallObstacle(0.5, 2.3, 2.6, 2.9),
            LocalWallObstacle(-0.2, 6.2, -0.2, 0.0),
            LocalWallObstacle(-0.2, 6.2, 6.0, 6.2),
            LocalWallObstacle(-0.2, 0.0, -0.2, 6.2),
            LocalWallObstacle(6.0, 6.2, -0.2, 6.2),
        ]
        return Environment(
            name="fallback_maze",
            bounds=((0.0, 0.0), (6.0, 6.0)),
            obstacles=walls,
            start=np.array([0.4, 0.4, 0.0]),
            goal=np.array([5.4, 4.2, -np.pi / 2]),
            robot_radius=0.1
        )


# =====================================================================
# 3. GRAPHICS INTERFACE RENDERERS FOR SYSTEM OBSTACLES
# =====================================================================

def draw_obstacle(surface: pygame.Surface, obstacle, to_screen, ppm: float):
    """
    Renders obstacles from external classes using coordinate reflection.
    Translates coordinate properties dynamically for duck-typed frameworks.
    """
    # Check if circle obstacle by attribute inspection
    if hasattr(obstacle, "cx") and hasattr(obstacle, "radius"):
        cx, cy, radius = obstacle.cx, obstacle.cy, obstacle.radius
        sx, sy = to_screen(cx, cy)
        screen_r = int(radius * ppm)
        pygame.draw.circle(surface, (150, 60, 60), (sx, sy), screen_r)
        pygame.draw.circle(surface, (220, 100, 100), (sx, sy), screen_r, 2)
        
    # Check if wall/box obstacle
    elif hasattr(obstacle, "xmin") and hasattr(obstacle, "xmax"):
        xmin, xmax = obstacle.xmin, obstacle.xmax
        ymin, ymax = obstacle.ymin, obstacle.ymax
        
        # Math-space top-left (xmin, ymax) and bottom-right (xmax, ymin)
        sx_tl, sy_tl = to_screen(xmin, ymax)
        sx_br, sy_br = to_screen(xmax, ymin)
        
        rect = pygame.Rect(sx_tl, sy_tl, sx_br - sx_tl, sy_br - sy_tl)
        pygame.draw.rect(surface, (80, 80, 80), rect)
        pygame.draw.rect(surface, (140, 140, 140), rect, 2)


# =====================================================================
# 4. SIMULATION RUNNER ENGINE
# =====================================================================

class Simulator2D:
    def __init__(self, env: Environment, kinematics: DiffDriveKinematics, qp_filter=None):
        self.env = env
        self.kinematics = kinematics
        self.qp_filter = qp_filter
        
        # Overwrite robot footprint dynamically to ensure it can pass gaps safely
        self.robot_radius = getattr(self.env, "robot_radius", 0.1)
        
        # Heatmap variables
        self.show_heatmap = False
        self.heatmap_surface = None
        self.last_heatmap_theta = 0.0
       
        # Collision notice variables
        self.collision_count = 0
        self.collision_flash_frames = 0
        self.FLASH_DURATION = 45

        # Current physical state of robot: [x, y, theta]
        self.state = np.array(env.start, dtype=float)
        
        # Continuous integration time-step
        self.dt = 0.05  # seconds
        self.stepper = self.kinematics.discrete_dynamics(dt=self.dt, method="rk4")
        
        # Scaling factor: pixels per meter (PPM)
        # We look at the bounds to scale nicely on an 1024x768 display
        (xmin, ymin), (xmax, ymax) = self.env.bounds
        self.ppm = min(1024 / (xmax - xmin), 768 / (ymax - ymin)) * 0.9
        
        # Initialize Pygame Screen
        pygame.init()
        self.width = int((xmax - xmin) * self.ppm) + 40
        self.height = int((ymax - ymin) * self.ppm) + 40
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"Differential-Drive Sim: {env.name}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.big_font = pygame.font.SysFont(None, 56)
        
        # Keyboard tuning variables
        self.target_v = 0.0
        self.target_w = 0.0
        
        # Maximum rates of target velocities
        self.max_v = 1.0      # m/s
        self.max_w = 0.75      # rad/s
        self.accel_v = 0.15   # acceleration increment
        self.accel_w = 0.25   # rotational acceleration increment

    def to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Transforms Cartesian math coordinates (y goes up) to Pygame screen coordinates (y goes down)."""
        (xmin, ymin), _ = self.env.bounds
        sx = int((x - xmin) * self.ppm) + 20
        sy = self.height - (int((y - ymin) * self.ppm) + 20)
        return sx, sy

    def _generate_heatmap(self):
        """Generates a semi-transparent overlay of the CBF safety values."""
        if self.qp_filter is None:
            return
            
        print("Regenerating CBF Heatmap overlay... (this may momentarily pause the sim)")
        self.heatmap_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        (xmin, ymin), (xmax, ymax) = self.env.bounds
        
        res = 0.15  # Grid cell size in meters (increased density for accuracy)
        theta = self.state[2]
        self.last_heatmap_theta = theta
        
        x_vals = np.arange(xmin, xmax, res)
        y_vals = np.arange(ymin, ymax, res)
        
        for x in x_vals:
            for y in y_vals:
                h = self.qp_filter.cbf.get_cbf_value(np.array([x, y, theta]))
                
                if h >= 0:
                    # Safe zone (Green)
                    alpha = min(160, int(h * 35 + 25))
                    color = (0, 255, 0, alpha)
                else:
                    # Danger zone (Red)
                    alpha = min(160, int(abs(h) * 55 + 65))
                    color = (255, 0, 0, alpha)
                    
                sx1, sy1 = self.to_screen(x, y + res)  # Screen top-left
                sx2, sy2 = self.to_screen(x + res, y)  # Screen bottom-right
                
                rect = pygame.Rect(sx1, sy1, sx2 - sx1, sy2 - sy1)
                pygame.draw.rect(self.heatmap_surface, color, rect)

    def check_collision(self, state: np.ndarray) -> bool:
        """Validates current coordinate bounds and obstacles collision."""
        (xmin, ymin), (xmax, ymax) = self.env.bounds
        x, y, _ = state
        
        # Boundary limits (accounting for robot footprint radius)
        if (x - self.robot_radius < xmin or x + self.robot_radius > xmax or
            y - self.robot_radius < ymin or y + self.robot_radius > ymax):
            return True
            
        # Specific obstacle collisions using their embedded `.collides` method
        pt = np.array([x, y])
        for obs in self.env.obstacles:
            if hasattr(obs, 'collides'):
                if obs.collides(pt, self.robot_radius):
                    return True
        return False

    def handle_input(self):
        """Processes key presses, scaling up/down [v, w] control values."""
        keys = pygame.key.get_pressed()
        
        # Linear Velocity input (v)
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.target_v = min(self.max_v, self.target_v + self.accel_v)
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.target_v = max(-self.max_v / 2, self.target_v - self.accel_v)
        else:
            # Active deceleration
            self.target_v *= 0.85
            if abs(self.target_v) < 0.05:
                self.target_v = 0.0
                
        # Angular Velocity input (omega)
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.target_w = min(self.max_w, self.target_w + self.accel_w)
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.target_w = max(-self.max_w, self.target_w - self.accel_w)
        else:
            # Revert steering rotation to center passively
            self.target_w *= 0.70
            if abs(self.target_w) < 0.05:
                self.target_w = 0.0

    def draw_robot(self):
        """Draws the differential drive robot to scale with a heading indicator."""
        x, y, theta = self.state
        r = self.robot_radius
        
        # Physical coordinates mapped to flipped screen space
        sx, sy = self.to_screen(x, y)
        sr = int(r * self.ppm)
        
        # 1. Main Circular Robot Body
        pygame.draw.circle(self.screen, (0, 130, 230), (sx, sy), sr)
        pygame.draw.circle(self.screen, (255, 255, 255), (sx, sy), sr, 2)
        
        # 2. Heading Indicator (Draw a line matching theta orientation)
        hx = x + r * np.cos(theta)
        hy = y + r * np.sin(theta)
        hsx, hsy = self.to_screen(hx, hy)
        pygame.draw.line(self.screen, (255, 190, 0), (sx, sy), (hsx, hsy), 3)
        pygame.draw.circle(self.screen, (255, 50, 50), (hsx, hsy), 4)

        # 3. Display left/right wheel silhouettes representing Diff-Drive config
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        wb = self.kinematics.robot.wheel_base
        
        # Left and Right Wheel Center coordinates in 2D plane
        lw_x = x - (wb / 2) * sin_t
        lw_y = y + (wb / 2) * cos_t
        rw_x = x + (wb / 2) * sin_t
        rw_y = y - (wb / 2) * cos_t
        
        # Simple line drawing representing tires
        t_half_len = 0.12 # physical wheel scale length
        for wx, wy in [(lw_x, lw_y), (rw_x, rw_y)]:
            p1x = wx - t_half_len * cos_t
            p1y = wy - t_half_len * sin_t
            p2x = wx + t_half_len * cos_t
            p2y = wy + t_half_len * sin_t
            
            pygame.draw.line(self.screen, (20, 20, 20), self.to_screen(p1x, p1y), self.to_screen(p2x, p2y), 6)

    def run(self):
        running = True
        while running:
            self.clock.tick(60) # Lock to 60hz frames
            
            # Standard quit checker & Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_h:
                        self.show_heatmap = not self.show_heatmap
                        if self.show_heatmap:
                            self._generate_heatmap()
                    elif event.key == pygame.K_r:
                        # Reset robot to start position
                        self.state = np.array(self.env.start, dtype=float)
                        self.target_v = 0.0
                        self.target_w = 0.0

            # Update controls via Key States
            self.handle_input()
            u = np.array([self.target_v, self.target_w])
            
            # --- APPLY QP FILTER FOR SAFETY ---
            if self.qp_filter is not None:
                # Get the safe control inputs that satisfy CBF constraints
                u = self.qp_filter.filter_control(self.state, u)
                
                # Reflect the safe controls back to our target variables 
                # so the user's manual input doesn't wind up infinitely
                self.target_v, self.target_w = float(u[0]), float(u[1])
            # ----------------------------------
            
            # Predict step using RK4 integrator from DiffDriveKinematics
            next_state = self.stepper(self.state, u, self.dt)
            
            # Check collisions with boundary or obstacles
            if not self.check_collision(next_state):
                self.state = next_state
            else:
                # Basic response: Stop linear speed entirely on collision
                self.target_v = 0.0
                self.collision_count += 1
                self.collision_flash_frames = self.FLASH_DURATION
                print(f"[COLLISION] Robot hit an obstacle/wall at "
                      f"({self.state[0]:.2f}, {self.state[1]:.2f}), "
                      f"heading {math.degrees(self.state[2]):.1f}° "
                      f"— total collisions: {self.collision_count}")


            # Render Logic
            self.screen.fill((30, 32, 38)) # Slate background
            
            # Draw Heatmap Overlay if toggled
            if self.show_heatmap and self.qp_filter is not None:
                # Refresh heatmap if heading changes drastically (CBF depends on theta)
                if self.heatmap_surface is None or abs(self.state[2] - self.last_heatmap_theta) > 0.4:
                    self._generate_heatmap()
                self.screen.blit(self.heatmap_surface, (0, 0))
            
            # Draw Start Pose
            sx, sy = self.to_screen(self.env.start[0], self.env.start[1])
            pygame.draw.circle(self.screen, (46, 125, 50), (sx, sy), 8, 2)
            
            # Draw Goal Target Point
            gx, gy = self.to_screen(self.env.goal[0], self.env.goal[1])
            pygame.draw.circle(self.screen, (198, 40, 40), (gx, gy), 12)
            pygame.draw.circle(self.screen, (255, 255, 255), (gx, gy), 6)
            
            # Draw Environment Obstacles dynamically (using duck-typed rendering)
            for obstacle in self.env.obstacles:
                draw_obstacle(self.screen, obstacle, self.to_screen, self.ppm)
                
            # Draw Robot Instance
            self.draw_robot()
            
            # Telemetry Display Text overlay
            cbf_val = 0.0
            if self.qp_filter is not None:
                cbf_val = self.qp_filter.cbf.get_cbf_value(self.state)
                
            font = pygame.font.SysFont(None, 24)
            speed_text = font.render(f"Linear Speed (v): {self.target_v:.2f} m/s", True, (240, 240, 240))
            yaw_text = font.render(f"Yaw Velocity (omega): {self.target_w:.2f} rad/s", True, (240, 240, 240))
            pos_text = font.render(f"State (X, Y, Theta): [{self.state[0]:.2f}, {self.state[1]:.2f}, {math.degrees(self.state[2]):.1f}°]", True, (200, 200, 200))
            
            # Draw dynamic CBF string
            cbf_color = (100, 255, 100) if cbf_val >= 0 else (255, 100, 100)
            cbf_text = font.render(f"CBF Value (h): {cbf_val:.4f}", True, cbf_color)
            
            control_text = font.render("Controls: Arrow keys / WASD | 'H' toggles Heatmap | 'R' to Reset" , True, (170, 170, 170))
            collisions_text = font.render(f"Collisions: {self.collision_count}", True, (255, 150, 150) if self.collision_count else (170, 170, 170))

            self.screen.blit(speed_text, (20, 20))
            self.screen.blit(yaw_text, (20, 45))
            self.screen.blit(pos_text, (20, 70))
            
            if self.qp_filter is not None:
                self.screen.blit(cbf_text, (20, 95))
            else:
                missing_cbf_text = font.render("CBF Value (h): [No QP Filter Loaded]", True, (150, 150, 150))
                self.screen.blit(missing_cbf_text, (20, 95))
                
            self.screen.blit(collisions_text, (20, 120))
            self.screen.blit(control_text, (20, self.height - 35))
            
            # --- COLLISION NOTICE OVERLAY ---
            if self.collision_flash_frames > 0:
                # Fade the flash out over its duration instead of an abrupt cut.
                fade = self.collision_flash_frames / self.FLASH_DURATION
                border_alpha = int(180 * fade)
 
                # Pulsing red border around the whole play area to draw the eye.
                border_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                border_thickness = 10
                pygame.draw.rect(border_surface, (255, 0, 0, border_alpha),
                                  border_surface.get_rect(), border_thickness)
                self.screen.blit(border_surface, (0, 0))
 
                # Centered "COLLISION!" banner.
                banner_text = self.big_font.render("COLLISION!", True, (255, 60, 60))
                banner_rect = banner_text.get_rect(center=(self.width // 2, 50))
                pygame.draw.rect(self.screen, (30, 32, 38), banner_rect.inflate(30, 16))
                self.screen.blit(banner_text, banner_rect)
 
                self.collision_flash_frames -= 1
            # ---------------------------------

            pygame.display.flip()

        pygame.quit()
        sys.exit()


# =====================================================================
# 5. EXECUTION & INITIALIZATION SETUP
# =====================================================================

if __name__ == "__main__":
    # 1. Instantiate robot physics configuration parameters
    # Wheelbase is scaled down slightly to visually fit the maze scale
    from environments.vehicle_dynamics import RobotModel, DiffDriveKinematics
    robot = RobotModel(wheel_radius=0.05, wheel_base=0.1)
    
    # 2. Setup kinematic constraints with our custom robot model
    kinematics = DiffDriveKinematics(robot)
    
    # --- SETUP CBF & QP FILTER ---
    # To activate the filter, ensure your eqx path is correct and uncomment:
    from sim.load_CBF import NNCBF
    from sim.qp_filter import QPFilter
    
    cbf_instance = NNCBF("runs/models/sweep_maze_separation/ls20_lu20/model.eqx")
    qp_filter_instance = QPFilter(cbf=cbf_instance, model=kinematics, alpha=1.0)
    
    # Placeholder for standalone execution
    # qp_filter_instance = None 
    # -----------------------------

    # 3. Create simulated world environment with boundaries and obstacles dynamically
    # Import path: environments.environment
    environment = build_env("maze")
    
    # Override robot_radius if necessary to cleanly traverse gaps in the loaded maze env
    environment.robot_radius = 0.1 
    
    # 4. Fire up the interactive simulation window
    sim = Simulator2D(environment, kinematics, qp_filter=qp_filter_instance)
    sim.run()
