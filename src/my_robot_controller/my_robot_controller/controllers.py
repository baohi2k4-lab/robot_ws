import math
import time
import numpy as np
from my_robot_controller.interfaces import ControllerInterface

class VelocityLimiter:
    """
    Decoupled utility to limit linear and angular speed and acceleration.
    """
    def __init__(self, max_linear_speed=10.0, max_angular_speed=10.0, 
                 max_linear_accel=50.0, max_angular_accel=50.0):
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.max_linear_accel = max_linear_accel
        self.max_angular_accel = max_angular_accel
        self.prev_linear = 0.0
        self.prev_angular = 0.0

    def limit(self, target_linear, target_angular, dt=0.067):
        # 1. Apply absolute velocity limits
        lim_linear = np.clip(target_linear, -self.max_linear_speed, self.max_linear_speed)
        lim_angular = np.clip(target_angular, -self.max_angular_speed, self.max_angular_speed)

        # 2. Apply acceleration limits if dt is valid
        if dt > 0.001:
            max_dlin = self.max_linear_accel * dt
            max_dang = self.max_angular_accel * dt
            lim_linear = np.clip(lim_linear, self.prev_linear - max_dlin, self.prev_linear + max_dlin)
            lim_angular = np.clip(lim_angular, self.prev_angular - max_dang, self.prev_angular + max_dang)

        self.prev_linear = lim_linear
        self.prev_angular = lim_angular
        return lim_linear, lim_angular

    def reset(self):
        self.prev_linear = 0.0
        self.prev_angular = 0.0


class TrackingControllerSMC(ControllerInterface):
    """
    Sliding Mode Controller for Lane Tracking.
    """
    def __init__(self):
        self.lambda_smc = 2.5
        self.k_smc = 4.2
        self.eta_smc = 0.8
        self.phi_smc = 0.4
        self.linear_speed = 0.10
        self.turn_angular_speed = 0.65
        self.prev_error = 0.0
        self.enabled = False

    def initialize(self, lambda_smc=2.5, k_smc=4.2, eta_smc=0.8, phi_smc=0.4, 
                   linear_speed=0.10, turn_angular_speed=0.65):
        self.lambda_smc = lambda_smc
        self.k_smc = k_smc
        self.eta_smc = eta_smc
        self.phi_smc = phi_smc
        self.linear_speed = linear_speed
        self.turn_angular_speed = turn_angular_speed
        self.prev_error = 0.0
        self.enabled = True

    def start(self):
        self.enabled = True

    def stop(self):
        self.enabled = False

    def reset(self):
        self.prev_error = 0.0

    def compute_command(self, smoothed_angle_deg, dt_actual=0.067):
        if not self.enabled:
            return {
                "linear_velocity": 0.0,
                "angular_velocity": 0.0,
                "status": "DISABLED",
                "controller_type": "SMC",
                "timestamp": time.time()
            }

        e = np.deg2rad(smoothed_angle_deg)
        dt_actual = np.clip(dt_actual, 0.001, 1.0)
        
        # Calculate derivative of error
        de = (e - self.prev_error) / dt_actual
        self.prev_error = e

        # Sliding surface
        S = de + self.lambda_smc * e
        sat_val = np.clip(S / self.phi_smc, -1.0, 1.0)
        
        # Tính mô-men quay góc dứt khoát
        angular_velocity = -self.k_smc * S - self.eta_smc * sat_val
        angular_velocity = np.clip(angular_velocity, -self.turn_angular_speed, self.turn_angular_speed)

        # Duy trì vận tốc tiến đều đặn và ổn định (linear_speed):
        # Không giảm đột ngột tốc độ khi rẽ, đảm bảo lực kéo của xe luôn ổn định
        # cả khi chạy thẳng lẫn khi quẹo trái/quẹo phải.
        return {
            "linear_velocity": self.linear_speed,
            "angular_velocity": angular_velocity,
            "status": "ACTIVE",
            "controller_type": "SMC",
            "timestamp": time.time()
        }

    def update_parameter(self, param_name, value):
        if hasattr(self, param_name):
            setattr(self, param_name, value)


class PurePursuitController(ControllerInterface):
    """
    Pure Pursuit Path Tracking Controller.
    """
    def __init__(self):
        self.turn_linear_speed = 0.07
        self.turn_angular_speed = 0.25
        self.path = []
        self.path_index = 0
        self.enabled = False

    def initialize(self, turn_linear_speed=0.07, turn_angular_speed=0.50):
        self.turn_linear_speed = turn_linear_speed
        self.turn_angular_speed = turn_angular_speed
        self.path = []
        self.path_index = 0
        self.enabled = True

    def start(self):
        self.enabled = True

    def stop(self):
        self.enabled = False

    def reset(self):
        self.path = []
        self.path_index = 0

    def set_path(self, path):
        if hasattr(path, 'waypoints'):
            self.path = path.waypoints
        elif isinstance(path, list):
            self.path = path
        else:
            self.path = []
        self.path_index = 0
        self.enabled = True

    def compute_command(self, current_x, current_y, current_yaw):
        if not self.enabled or not self.path:
            return {
                "linear_velocity": 0.0,
                "angular_velocity": 0.0,
                "status": "COMPLETED",
                "controller_type": "PurePursuit",
                "timestamp": time.time()
            }

        if self.path_index >= len(self.path):
            return {
                "linear_velocity": 0.0,
                "angular_velocity": 0.0,
                "status": "COMPLETED",
                "controller_type": "PurePursuit",
                "timestamp": time.time()
            }

        # 1. Find closest waypoint forward on path
        closest_idx = self.path_index
        min_d = float('inf')
        search_limit = min(len(self.path), self.path_index + 40)
        for i in range(self.path_index, search_limit):
            px, py = self.path[i]
            d = math.sqrt((px - current_x)**2 + (py - current_y)**2)
            if d < min_d:
                min_d = d
                closest_idx = i
        self.path_index = closest_idx

        # 2. Project lookahead point forward from closest waypoint (L_d = 0.28m)
        L_d = getattr(self, 'lookahead_dist', 0.28)
        target_idx = closest_idx
        for i in range(closest_idx, len(self.path)):
            px, py = self.path[i]
            d = math.sqrt((px - current_x)**2 + (py - current_y)**2)
            if d >= L_d:
                target_idx = i
                break
        else:
            target_idx = len(self.path) - 1

        target = self.path[target_idx]
        dx = target[0] - current_x
        dy = target[1] - current_y
        dist = math.sqrt(dx*dx + dy*dy)
            
        if (self.path_index >= len(self.path) - 3 and dist < 0.35) or (self.path_index >= len(self.path) - 1 and dist < 0.50):
            return {
                "linear_velocity": 0.0,
                "angular_velocity": 0.0,
                "status": "COMPLETED",
                "controller_type": "PurePursuit",
                "timestamp": time.time()
            }
            
        target_yaw = math.atan2(dy, dx)
        yaw_error = target_yaw - current_yaw
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))

        # Skid-Steer Heading Alignment with Rolling Pivot (Chống cày đất đứng yên)
        abs_err = abs(yaw_error)
        min_roll = min(0.06, self.turn_linear_speed)
        if abs_err > math.radians(45.0):
            # Góc lệch cực lớn: bò tiến giữ đà phá ma sát tĩnh + quay vi sai mô-men cực đại
            linear_vel = min(0.055, self.turn_linear_speed)
            angular_vel = float(math.copysign(self.turn_angular_speed, yaw_error))
        elif abs_err > math.radians(20.0):
            # Góc lệch trung bình (20° - 45°): bò tiến 80% tốc độ ôm cua kết hợp bẻ lái mạnh
            linear_vel = float(np.clip(self.turn_linear_speed * 0.85, min_roll, self.turn_linear_speed))
            angular_vel = float(math.copysign(self.turn_angular_speed, yaw_error))
        else:
            # Thân xe đã bám sát tiếp tuyến (lệch <= 20°): Lăn bánh tiến êm mượt đúng 0.07 m/s bám quỹ đạo
            linear_vel = float(np.clip(self.turn_linear_speed * math.cos(yaw_error), min_roll, self.turn_linear_speed))
            angular_vel = 2.0 * linear_vel * math.sin(yaw_error) / max(0.20, dist)
            angular_vel = float(np.clip(angular_vel, -self.turn_angular_speed, self.turn_angular_speed))
        
        return {
            "linear_velocity": linear_vel,
            "angular_velocity": angular_vel,
            "status": "ACTIVE",
            "controller_type": "PurePursuit",
            "timestamp": time.time()
        }

    def update_parameter(self, param_name, value):
        if hasattr(self, param_name):
            setattr(self, param_name, value)


class SafetyController:
    """
    Safety Controller to handle Emergency Stop, collision warnings,
    and absolute speed restrictions.
    """
    def __init__(self, max_safe_linear=1.5, max_safe_angular=3.0, collision_stop_dist=0.35):
        self.max_safe_linear = max_safe_linear
        self.max_safe_angular = max_safe_angular
        self.collision_stop_dist = collision_stop_dist
        self.emergency_active = False

    def check_safety(self, linear, angular, obstacle_dist=None):
        """
        Overrides velocities if safety condition is violated.
        Returns (safe_linear, safe_angular, status).
        """
        if self.emergency_active:
            return 0.0, 0.0, "EMERGENCY_STOP"

        if obstacle_dist is not None and obstacle_dist < self.collision_stop_dist:
            self.emergency_active = True
            return 0.0, 0.0, "EMERGENCY_STOP"

        # Apply absolute safety cap
        safe_linear = np.clip(linear, -self.max_safe_linear, self.max_safe_linear)
        safe_angular = np.clip(angular, -self.max_safe_angular, self.max_safe_angular)

        return safe_linear, safe_angular, "SAFE"

    def set_emergency(self, active):
        self.emergency_active = active


class ControllerManager:
    """
    Manager to orchestrate and switch between active controllers.
    """
    def __init__(self, limiter=None, safety=None):
        self.controllers = {}
        self.active_name = None
        self.limiter = limiter if limiter is not None else VelocityLimiter()
        self.safety = safety if safety is not None else SafetyController()

    def register_controller(self, name, controller):
        self.controllers[name] = controller

    def select_controller(self, name):
        if name in self.controllers:
            if self.active_name != name:
                if self.active_name in self.controllers:
                    self.controllers[self.active_name].stop()
                self.active_name = name
                self.controllers[name].start()
            return True
        return False

    def get_active_controller(self):
        if self.active_name:
            return self.controllers[self.active_name]
        return None

    def compute_command(self, *args, **kwargs):
        """
        Calculates command from active controller, applies velocity limiter, 
        and passes it through the safety controller.
        """
        active = self.get_active_controller()
        if not active:
            return {
                "linear_velocity": 0.0,
                "angular_velocity": 0.0,
                "status": "NO_ACTIVE_CONTROLLER",
                "controller_type": "NONE",
                "timestamp": time.time()
            }

        # Check if obstacle distance is passed in kwargs for safety checks
        obstacle_dist = kwargs.pop('obstacle_dist', None)
        dt = kwargs.get('dt_actual', 0.067)

        # Compute raw output from controller
        res = active.compute_command(*args, **kwargs)
        
        raw_linear = res["linear_velocity"]
        raw_angular = res["angular_velocity"]

        # 1. Apply velocity/acceleration limit
        lim_linear, lim_angular = self.limiter.limit(raw_linear, raw_angular, dt)

        # 2. Pass through safety controller
        safe_linear, safe_angular, safety_status = self.safety.check_safety(
            lim_linear, lim_angular, obstacle_dist
        )

        status = safety_status if safety_status != "SAFE" else res["status"]

        return {
            "linear_velocity": safe_linear,
            "angular_velocity": safe_angular,
            "status": status,
            "controller_type": res["controller_type"],
            "timestamp": time.time()
        }

    def reset(self):
        if self.limiter:
            self.limiter.reset()
        for c in self.controllers.values():
            c.reset()
