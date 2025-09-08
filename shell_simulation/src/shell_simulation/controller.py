import numpy as np
import math
from collections import deque
import time
from enum import Enum
import csv
import tf
from geometry_msgs.msg import TransformStamped


# ==========================
# Vehicle Physics Constants
# ==========================
Cd = 0.15000000596           # Drag coefficient
m = 1845.0                   # Vehicle mass (kg)
g = 9.81                     # Gravity (m/s^2)
rho = 1.2                    # Air density (kg/m^3)
A = 2.22                     # Frontal area (m^2)
Crr = 0.01                   # Rolling resistance coefficient
THROTTLE_CAP = 0.4

# ==========================
# Utility Functions
# ==========================
def euler_from_quaternion(quat):
    x, y, z, w = quat
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z

class CurveSpeedAdapter:
    def __init__(self, min_speed=25.0, max_speed=40):
        self.MIN_SPEED = min_speed
        self.MAX_SPEED = max_speed
        self.last_target_speed = max_speed
        self.curve_start_distance = 50
        self.curve_end_distance = 15
        self.emergency_brake_threshold = 0.4
        self.base_curve_threshold = 0.06
        self.coast_start_distance = 25
        self.coast_intensity = 0.7
        self.burn_intensity_factor = 0.2
        self.curve_look_ahead_short = 9
        self.curve_look_ahead_medium = 18
        self.curve_look_ahead_long = 35
        self.approaching_curve = False
        self.in_curve = False
        self.exiting_curve = False
        self.curve_distance = float('inf')
        self.curve_exit_distance = float('inf')
        self.curve_severity = 0.0
        self.test = False

    def detect_sharp_curve(self, waypoints, start_idx, lookahead=2, angle_threshold_deg=10.0):
        """Scans ahead starting from (start_idx + lookahead) for sharp curves."""
        deg_threshold_rad = np.radians(angle_threshold_deg)

        end_idx = min(start_idx + lookahead + 20, len(waypoints))  # optional extra safety (+20)

        # Start at start_idx + lookahead, not at start_idx
        for i in range(start_idx + lookahead, end_idx - 5):
            x1, y1 = waypoints[i]
            x2, y2 = waypoints[i + 3]
            x3, y3 = waypoints[i + 5]

            vec1 = np.array([x2 - x1, y2 - y1])
            vec2 = np.array([x3 - x2, y3 - y2])

            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 < 1e-3 or norm2 < 1e-3:
                continue

            cos_angle = np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0)
            angle = np.arccos(cos_angle)

            if angle > deg_threshold_rad:
                self.test = True
                return True

        return False

# ==============================================
# Enhanced PID Controller
# ==============================================
class EnhancedPIDLongitudinalController:
    def __init__(self, curve_adapter, K_P=0.15, K_I=0.001, K_D=0.01, dt=0.03):
        self.curve_adapter = curve_adapter
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._error_buffer = deque([0.0]*10, maxlen=10)
        self._max_speed = 40
        self._last_throttle = 0.0
        self._min_throttle = 0.2
        self._max_throttle = 0.4
        self._speed_tolerance = 0.7
        self.waypoints = None
        self._last_speeds = deque(maxlen=20)
        self._last_accel_time = time.time()
        self._target_speed = 0.0
        self._throttle_smoothing_factor = 0.8
        self._last_location = None
        self._min_speed_threshold = 25.0
        self._last_throttle_change_time = time.time()
        self._min_throttle_change_interval = 0.12
        self._current_speed = 0.0
        self._current_position = None
        self._current_orientation = None
        self._accel_ramp_up_rate = 0.006
        self._accel_ramp_down_rate = 0.004
        self._curve_decay_active = False
        self._curve_decay_start_time = None
        self._initial_throttle_at_decay_start = 0.0

    def update_state(self, odometry_msg, speed):
        self._current_speed = speed
        self._current_position = odometry_msg.pose.pose.position
        self._current_orientation = odometry_msg.pose.pose.orientation

    def update_waypoints(self, waypoints):
        self.waypoints = waypoints

    def estimate_throttle_for_speed(self, speed_kmh):
        v = speed_kmh / 3.6  # convert to m/s
        drag = 0.5 * rho * Cd * A * v**2
        rolling = m * g * Crr
        total_force = drag + rolling
        required_acc = total_force / m
        base_throttle = required_acc  # assuming linear map: 1 throttle ≈ 1 m/s²

        return np.clip(base_throttle, 0.08, self._max_throttle)
    
    def _calculate_throttle(self, target_speed, current_speed):
        # Decay logic based on speed, not time
        if self._curve_decay_active:
            if current_speed > (21.0 / 3.6):  # 20 km/h in m/s
                throttle = self._last_throttle - 0.04  # decay throttle slightly every cycle
                throttle = max(0.05, throttle)  # don't let it go to 0
                self._last_throttle = throttle  # update last throttle to decay progressively
                return throttle
            else:
                # If speed dropped to <=20 km/h, stop decaying
                self._curve_decay_active = False

        # Normal throttle control below
        # Your PID logic continues here:
        speed_error = target_speed - current_speed
                # Anti-windup reset if speed error changes sign
        if len(self._error_buffer) > 0 and speed_error * self._error_buffer[-1] < 0:
            self._error_buffer.clear()
        self._error_buffer.append(speed_error)
        self._target_speed = target_speed

        if abs(speed_error) < self._speed_tolerance:
            new_throttle = self._last_throttle
        elif speed_error > 0:
            new_throttle = self._last_throttle + self._accel_ramp_up_rate
        else:
            # Estimate the throttle needed for the target speed
            estimated_target_throttle = self.estimate_throttle_for_speed(target_speed)
    
            # Decay quickly but never go below estimated needed throttle
            fast_decay = 0.015  # Faster ramp down than before
            new_throttle = max(self._last_throttle - fast_decay, estimated_target_throttle)

        if len(self._error_buffer) > 0:
            pid_correction = np.clip(
                (self._k_p * speed_error +
                 self._k_i * sum(self._error_buffer) * self._dt +
                 self._k_d * (speed_error - self._error_buffer[0]) / self._dt) * 0.0008,
                -0.02, 0.02
            )
            new_throttle += pid_correction

        new_throttle = np.clip(new_throttle, 0.0, self._max_throttle)
        smoothed_throttle = (self._throttle_smoothing_factor * self._last_throttle +
                            (1 - self._throttle_smoothing_factor) * new_throttle)

        if (abs(smoothed_throttle - self._last_throttle) > 0.005 or
            time.time() - self._last_throttle_change_time > self._min_throttle_change_interval):
            self._last_throttle = smoothed_throttle
            self._last_accel_time = time.time()
            self._last_throttle_change_time = time.time()

        return self._last_throttle
        
    def run_step(self):
        if not self.waypoints:
            throttle = self._calculate_throttle(self._max_speed, self._current_speed)
            self._last_speeds.append(self._current_speed)
            return throttle

        curr_idx = self.find_nearest_waypoint_index()
        sharp_curve_detected = self.curve_adapter.detect_sharp_curve(
            self.waypoints,
            start_idx=curr_idx,
            lookahead=30,
            angle_threshold_deg=6.0  # or whatever angle threshold you want
        )

        if sharp_curve_detected:
            if not self._curve_decay_active and self._current_speed > (21.0 / 3.6):
                self._curve_decay_active = True
                self._curve_decay_start_time = time.time()
                self._initial_throttle_at_decay_start = self._last_throttle
                print(f"[DEBUG] Sharp curve detected at idx {curr_idx}, starting decay (speed={self._current_speed:.2f} km/h)")
            else:
                print(f"[DEBUG] Sharp curve detected but speed={self._current_speed:.2f} km/h, no decay triggered.")
            target_speed = min(self._max_speed, self._target_speed)  # Use the lower of max_speed and target_speed
        else:
            target_speed = self._target_speed  # Use the target speed from vehicle controller

        if target_speed < self._min_speed_threshold and self._current_speed <= self._min_speed_threshold:
            target_speed = self._min_speed_threshold

        throttle = self._calculate_throttle(target_speed, self._current_speed)
        self._last_speeds.append(self._current_speed)
        return throttle

    def update_waypoints(self, waypoints):
        self.waypoints = waypoints

    def find_nearest_waypoint_index(self):
        if not self._current_position:
            return 0
        curr_xy = np.array([self._current_position.x, self._current_position.y])
        waypoints_xy = np.array([(wp[0], wp[1]) for wp in self.waypoints])
        return np.argmin(np.sum((curr_xy - waypoints_xy)**2, axis=1))

# ==============================================
# Pure Pursuit Controller
# ==============================================
class PurePursuitController:
    def __init__(self, wheelbase, min_lookahead=5.0, max_lookahead=15.0):
        self.WB = wheelbase
        self.MIN_LOOKAHEAD = min_lookahead
        self.MAX_LOOKAHEAD = max_lookahead
        self.last_target_idx = 0
        self.last_steering = 0.0
        self.steering_history = deque(maxlen=10)
        self.steering_oscillation_count = 0
        self.max_oscillation_count = 5
        self.in_sharp_curve = False
        self.last_sharp_curve_time = 0.0  # Timestamp when sharp curve was last detected

    def find_nearest_waypoint(self, waypoints, xc, yc):
        curr_xy = np.array([xc, yc])
        waypoints_xy = np.array([(wp[0], wp[1]) for wp in waypoints])
        nearest_idx = np.argmin(np.sum((curr_xy - waypoints_xy)**2, axis=1))
        return nearest_idx

    def idx_close_to_lookahead(self, waypoints, idx, xc, yc, lookahead):
        best_idx = idx
        min_distance_diff = float('inf')
        for i in range(idx, len(waypoints)):
            dist = self.find_distance_index_based(waypoints, i, xc, yc)
            if dist <= (lookahead):  # only consider waypoints within (lookahead - 1) meters
                distance_diff = abs(dist - lookahead)
                if distance_diff < min_distance_diff:
                    min_distance_diff = distance_diff
                    best_idx = i
            else:
                break  # stop once we go beyond (lookahead - 1)
        return best_idx

    def find_distance_index_based(self, waypoints, idx, xc, yc):
        x1, y1 = float(waypoints[idx][0]), float(waypoints[idx][1])
        return math.sqrt((x1 - xc)**2 + (y1 - yc)**2)

    def pure_pursuit(self, waypoints, xc, yc, yaw, vel):
        if not waypoints:
            return 0.0

        # Hugging Curves Better
        current_time = time.time()
        time_since_last_curve = current_time - self.last_sharp_curve_time

        current_time = time.time()
        time_since_last_curve = current_time - self.last_sharp_curve_time

        if self.in_sharp_curve or time_since_last_curve < 3.0:
            # Sharp curve or just exited sharp curve recently -> smaller lookahead
            lookahead_dist = np.clip(vel * 0.2 + 3.0, 4.0, 7.0)  # Smaller, tighter
        else:
            # Normal driving
            lookahead_dist = np.clip(vel * 0.3 + 5.0, self.MIN_LOOKAHEAD, self.MAX_LOOKAHEAD)


        # Always find the nearest waypoint
        nearest_idx = self.find_nearest_waypoint(waypoints, xc, yc)

        # Search for the first waypoint within (lookahead - 1) meters
        best_idx = nearest_idx
        min_diff = float('inf')

        for i in range(nearest_idx, len(waypoints)):
            dx = waypoints[i][0] - xc
            dy = waypoints[i][1] - yc
            dist = math.hypot(dx, dy)

            if dist <= (lookahead_dist):
                diff = abs(dist - lookahead_dist)
                if diff < min_diff:
                    min_diff = diff
                    best_idx = i
            else:
                break  # stop immediately if waypoint is farther

        target_idx = best_idx
        target_x, target_y = waypoints[target_idx][0], waypoints[target_idx][1]

        dx = target_x - xc
        dy = target_y - yc
        distance_to_target = math.hypot(dx, dy)



        if distance_to_target > (lookahead_dist):
            return 0.0

        # Calculate steering
        target_angle = math.atan2(dy, dx)
        alpha = (target_angle - yaw + math.pi) % (2 * math.pi) - math.pi


        # Compute steering angle
        steering_angle = math.atan2(2 * self.WB * math.sin(alpha), lookahead_dist)
        steering_angle = max(-math.radians(70), min(math.radians(70), steering_angle))  # Clamp
        normalized_steering = -steering_angle / math.radians(70)

        self.last_steering = normalized_steering  # Store for energy penalty logic
        self.steering_history.append(normalized_steering)
        smoothed_steering = sum(self.steering_history) / len(self.steering_history)
        self.last_steering = smoothed_steering
        return smoothed_steering

# ==============================================
# Vehicle Controller
# ==============================================
class ControlMode(Enum):
    STOP = "stop"
    PROCEED = "proceed"
    FOLLOW = "follow"

class VehicleController:
    def __init__(self, node, wheelbase=2.8):
        self.node = node
        self.wheelbase = wheelbase
        
        # Initialize TF listener
        self._tf_listener = tf.TransformListener()
        
        # Initialize controllers
        self.curve_adapter = CurveSpeedAdapter()
        self.longitudinal_controller = EnhancedPIDLongitudinalController(self.curve_adapter)
        self.lateral_controller = PurePursuitController(wheelbase)
        
        # Initialize state variables
        self._current_position = None
        self._current_orientation = None
        self._current_speed = 0.0
        self._current_target_speed = 0.0
        self._current_target_speed_override = None
        self._current_mode = ControlMode.STOP
        self._waypoints = []
        self._stop_distance = 0.0
        self._last_control_time = time.time()
        self._control_dt = 0.03  # 30ms control loop
        self._curve_detected = False
        self._start_position = None
        self._file_path = None
        self._file_pointer = None
        self._batch_size = 100
        self._last_loaded_index = 0
        self.target_speed_override = None  # km/h
        self.speed_override_active = False
        self.pid_integral = 0.0
        self.pid_prev_error = 0.0
        self.prev_throttle_override = 0.0
        self.override_pid_dt = 0.05
        self._initial_pos_recorded = False
        self._initial_pos = None
        self._stopped = False   
        self._stop_start_position = None
        self._cumulative_moved_distance = 0.0
        self._last_stop_location = None
        self._gradual_start_throttle = 0.0
        self._current_target_speed_override = 0.0
        self._target_speed_ramp_rate = 1.0  # km/h per update cycle (adjust as needed)
        self._initial_throttle_at_start = None
        self._throttle_blocked_during_stop = False
        self._proceeding_after_stop = False
        self._throttle_ramp_after_stop = 0.0
        self._total_energy_joules = 0.0
        self._last_energy_timestamp = time.time()
        self._last_energy_speed = 0.0
        self._last_energy_position = None
        self._energy_log = []  # Optional: store tuples of (timestamp, speed, energy_kWh)
        self._throttle_locked_at_max = False
        self._throttle_cut_once_in_stop = False  # 🔒 permanent STOP throttle cut flag
        self._proceed_start_time = None  # Track when PROCEED/FOLLOW mode started

    def __del__(self):
        # Clean up TF listener
        if hasattr(self, '_tf_listener'):
            self._tf_listener = None

    def update_energy_usage(self, acceleration=0.0):
        if self._current_position is None:
            return 0.0  # Return 0 if not enough data

        now = time.time()
        dt = now - self._last_energy_timestamp
        if dt <= 0.01:
            return 0.0  # Too small to consider

        # Distance traveled
        if self._last_energy_position:
            dx = self._current_position.x - self._last_energy_position.x
            dy = self._current_position.y - self._last_energy_position.y
            distance = math.hypot(dx, dy)
        else:
            distance = 0.0

        v = self._current_speed / 3.6  # km/h to m/s
        theta = 0.0  # Assuming flat terrain for now
        a = acceleration

        # Optimized force calculation
        # 1. Reduced rolling resistance coefficient
        Crr_optimized = Crr * 0.8  # 20% reduction in rolling resistance
        
        # 2. Optimized drag coefficient
        Cd_optimized = Cd * 0.9  # 10% reduction in drag
        
        # 3. Reduced frontal area
        A_optimized = A * 0.95  # 5% reduction in frontal area

        # Penalty for steering effort → increases rolling resistance
        steering_penalty = abs(self.lateral_controller.last_steering) if hasattr(self.lateral_controller, "last_steering") else 0.0
        penalty_multiplier = 1.0 + 0.3 * steering_penalty  # up to +30% for full steering

        # Apply penalty to rolling resistance only
        Crr_penalized = Crr_optimized * penalty_multiplier
        
        # Total force with optimizations
        F = (
            m * g * Crr_penalized * math.cos(theta) +  # Penalized rolling resistance
            0.5 * rho * Cd_optimized * A_optimized * v**2 +  # Reduced drag
            m * a +  # Acceleration force
            m * g * math.sin(theta)  # Gravity force
        )

        # Instantaneous energy for this step (in Joules)
        energy_joules = F * distance
        energy_kwh = energy_joules / 3600000  # 1 kWh = 3.6 million J

        # Cumulative tracking
        self._total_energy_joules += energy_joules
        self._last_energy_timestamp = now
        self._last_energy_position = self._current_position
        self._energy_log.append((now, self._current_speed, energy_kwh))

        return energy_kwh
        
    def _override_throttle_control(self):
        # ✅ Always lock throttle at 0.4 in PROCEED mode if no curve
        if self._current_mode == ControlMode.PROCEED and not self.lateral_controller.in_sharp_curve:
            if not self._throttle_locked_at_max:
                pass
            self._throttle_locked_at_max = True
            self.prev_throttle_override = THROTTLE_CAP
            return THROTTLE_CAP

        # ⏩ Continue normal override logic below (for curves or other modes)
        if self.target_speed_override is not None:
            if self._current_target_speed_override < self.target_speed_override:
                self._current_target_speed_override = min(
                    self._current_target_speed_override + self._target_speed_ramp_rate,
                    self.target_speed_override
                )
            elif self._current_target_speed_override > self.target_speed_override:
                self._current_target_speed_override = max(
                    self._current_target_speed_override - self._target_speed_ramp_rate,
                    self.target_speed_override
                )
        else:
            self._current_target_speed_override = 0.0

        if self.target_speed_override == 0.0:
            return 0.0

        current_speed_mps = self._current_speed / 3.6  # km/h to m/s
        target_speed_mps = self._current_target_speed_override / 3.6
        speed_error = target_speed_mps - current_speed_mps
        speed_error_kmh = (target_speed_mps - current_speed_mps) * 3.6

        # PID Controller Parameters
        Kp = 0.20
        Ki = 0.001
        Kd = 0.005
        dt = self.override_pid_dt

        # Anti-windup reset
        if speed_error * self.pid_prev_error < 0:
            self.pid_integral = 0.0

        # PID Computation
        self.pid_integral += speed_error * dt
        derivative = (speed_error - self.pid_prev_error) / dt
        pid_output = (Kp * speed_error) + (Ki * self.pid_integral) + (Kd * derivative)

        # Base throttle and ramp depending on target speed
        if self._current_target_speed_override <= 10.0 / 3.6:
            base_throttle = 0.10 + 0.015 * np.clip(current_speed_mps / 5.0, 0.0, 1.0)
            ramp_rate = 0.008
        elif self._current_target_speed_override < 18.0 / 3.6:
            base_throttle = 0.18 + 0.02 * np.clip(current_speed_mps / 10.0, 0.0, 1.0)
            ramp_rate = 0.006
        elif self._current_target_speed_override >= 25.0 / 3.6:
            base_throttle = 0.34 + 0.026 * np.clip(current_speed_mps / 30.0, 0.0, 1.0)
            ramp_rate = 0.004
        else:
            base_throttle = 0.31 + 0.020 * np.clip(current_speed_mps / 25.0, 0.0, 1.0)
            ramp_rate = 0.0024

        desired_throttle = base_throttle + pid_output

        # Lock throttle permanently at 0.4 after ramp-up in PROCEED
        if self._current_mode == ControlMode.PROCEED:
            if not self._throttle_locked_at_max:
                # Smooth ramp-up until 0.4
                if desired_throttle >= THROTTLE_CAP:
                    self._throttle_locked_at_max = True
                    self.prev_throttle_override = THROTTLE_CAP
                    return THROTTLE_CAP
            else:
                # Once locked, hold exactly 0.4 without fluctuation
                return THROTTLE_CAP

        # Smooth ramp
        ramp_rate = 0.0012
        throttle_change = desired_throttle - self.prev_throttle_override
        if throttle_change > ramp_rate:
            throttle = self.prev_throttle_override + ramp_rate
        elif throttle_change < -ramp_rate:
            throttle = self.prev_throttle_override - ramp_rate
        else:
            throttle = desired_throttle

        self.prev_throttle_override = throttle
        self.pid_prev_error = speed_error

        # Define ramp parameters
        RAMP_UP_STEP = 0.0012     # Slower ramp-up (smaller = slower)
        RAMP_DOWN_STEP = 0.04     # Fast decay
        
        if current_speed_mps > target_speed_mps + 0.2:
            # Estimate the target throttle for the override speed
            estimated_target_throttle = 0.01 * (target_speed_mps * 3.6)  # convert to km/h for scale
            estimated_target_throttle = np.clip(estimated_target_throttle, 0.08, THROTTLE_CAP)

            # Decay toward target without undershooting
            raw_decay = max(self.prev_throttle_override - RAMP_DOWN_STEP, estimated_target_throttle)

            # Apply ramp-up smoothing even during decay
            if raw_decay < self.prev_throttle_override:
                throttle = raw_decay
            else:
                throttle = min(raw_decay, self.prev_throttle_override + RAMP_UP_STEP)
        else:
            # Normal throttle ramping logic
            throttle_change = desired_throttle - self.prev_throttle_override
            if throttle_change > RAMP_UP_STEP:
                throttle = self.prev_throttle_override + RAMP_UP_STEP
            elif throttle_change < -RAMP_DOWN_STEP:
                throttle = self.prev_throttle_override - RAMP_DOWN_STEP
            else:
                throttle = desired_throttle

        throttle = min(throttle, THROTTLE_CAP)
        return throttle

    def set_speed_override(self, target_speed_kmh):
        self.target_speed_override = target_speed_kmh
        self.speed_override_active = True

    def clear_speed_override(self):
        self.target_speed_override = None
        self.speed_override_active = False

    def stopping_control_logic(self, current_speed_mps, remaining_distance_m):
        """
        Calculate throttle, brake, and action for safe stopping.
        """
        vehicle_mass = 1845.0
        coasting_decel = -3.3  # m/s²
        max_throttle = 0.4
        ramp_rate = 0.0025

        # 🔒 Cut throttle completely and apply full brakes if stop distance < 5m
        if remaining_distance_m < 5.0:
            throttle = 0.0
            brake = 1.0
            self._throttle_blocked_during_stop = True
            self._throttle_cut_once_in_stop = True
            action = "Emergency Stop: Full Brake"
            return throttle, brake, action

        if remaining_distance_m <= 10.0 and current_speed_mps > (20.0 / 3.6):
            throttle = 0.0
            brake = 0.0
            action = "Near STOP @ High Speed → Coasting (no throttle/brake)"
            return throttle, brake, action
        
        # 🔒 Absolutely prevent throttle reapplication once cut
        if self._throttle_cut_once_in_stop:
            throttle = 0.0
            if remaining_distance_m <= 2.0:
                brake = 1.0
                action = "PERMA-LOCK: Full Stop"
            else:
                brake = 0.0
                action = "PERMA-LOCK: Coasting"
            return throttle, brake, action

        # 🚨 Emergency brake in overshoot case
        if self._cumulative_moved_distance >= self._stop_distance and current_speed_mps > 0.5:
            throttle = 0.0
            brake = 1.0
            self._throttle_blocked_during_stop = True
            self._throttle_cut_once_in_stop = True
            action = "Overshoot Emergency Brake"
            return throttle, brake, action

        # 🟢 Initial ramp-up phase
        if current_speed_mps < 1.50 and remaining_distance_m >= 5.0:
            target_speed = (2 * abs(coasting_decel) * remaining_distance_m) ** 0.5
            target_speed = min(target_speed, 8.0)

            if current_speed_mps < target_speed:
                self._gradual_start_throttle = min(self._gradual_start_throttle + ramp_rate, max_throttle)
                throttle = self._gradual_start_throttle
                brake = 0.0
                action = "Gradual Start Ramp"
            else:
                self._throttle_blocked_during_stop = True  # lock out throttle
                self._throttle_cut_once_in_stop = True
                throttle = 0.0
                brake = 1.0
                action = "Ramp Aborted - Brake and Hold"

        # ⚪ Coasting phase
        elif remaining_distance_m > 2.0:
            throttle = 0.0
            brake = 0.0
            action = "Coasting"

        # 🔴 Final brake zone
        else:
            throttle = 0.0
            brake = 1.0  # Apply full brakes in final zone
            self._throttle_blocked_during_stop = True
            self._throttle_cut_once_in_stop = True
            action = "Final Zone: Full Brake"

        return throttle, brake, action

    def update_odometry(self, odometry_msg):
        self._current_position = odometry_msg.pose.pose.position
        self._current_orientation = odometry_msg.pose.pose.orientation

        # Save start position once
        if not hasattr(self, '_start_position') or self._start_position is None:
            self._start_position = self._current_position

        # Save last print time once
        if not hasattr(self, '_last_print_time'):
            self._last_print_time = time.time()

        # Calculate moved distance
        dx = self._current_position.x - self._start_position.x
        dy = self._current_position.y - self._start_position.y
        moved_distance = math.hypot(dx, dy)

        # Print every 1 second
        current_time = time.time()
        if current_time - self._last_print_time >= 1.0:
            self._last_print_time = current_time

        # Convert quaternion to euler angles
        try:
            transform = TransformStamped()
            transform.transform.rotation = self._current_orientation

            euler = euler_from_quaternion([
                self._current_orientation.x,
                self._current_orientation.y,
                self._current_orientation.z,
                self._current_orientation.w
            ])

            self._current_yaw = euler[2]
            self._current_yaw = (self._current_yaw + math.pi) % (2 * math.pi) - math.pi
        except Exception as e:
            self._current_yaw = 0.0

        # Update longitudinal controller state
        self.longitudinal_controller.update_state(odometry_msg, self._current_speed)

    def update_imu(self, imu_msg):
        # Store IMU data if needed for future use
        pass

    def update_speed(self, speed):
        self._current_speed = speed

    def set_mode(self, mode: ControlMode, **kwargs):
        global THROTTLE_CAP
        self._throttle_blocked_during_stop = False  # reset block flag
        if mode != ControlMode.STOP:
            self._throttle_cut_once_in_stop = False

        # Reset throttle lock only when changing mode
        if mode != self._current_mode:
            self._throttle_locked_at_max = False
        
        # Detect STOP → PROCEED/FOLLOW transition
        if self._current_mode == ControlMode.STOP and mode in (ControlMode.PROCEED, ControlMode.FOLLOW):
            self._proceeding_after_stop = True
            self._proceed_start_time = time.time()

        if mode == ControlMode.STOP:
            THROTTLE_CAP = 0.4
            self._gradual_start_throttle = 0.0
            self._current_mode = mode
            self._stop_distance = kwargs.get('stop_distance', None)
        elif mode in (ControlMode.PROCEED, ControlMode.FOLLOW):
            THROTTLE_CAP = 0.4
            self._current_mode = mode
            self._current_target_speed = 40.0  # Fixed target speed for both modes

    def step_go(self):
        if not self._waypoints:
            return 0.0, 0.0, 0.0

        current_time = time.time()
        dt = current_time - self._last_control_time
        self._last_control_time = current_time

        if self._current_mode == ControlMode.STOP and self._stop_distance is not None:
            # --- Stopping Mode ---
            moved_distance = 0.0
            if self._last_stop_location and self._current_position:
                dx = self._current_position.x - self._last_stop_location.x
                dy = self._current_position.y - self._last_stop_location.y
                incremental_distance = math.hypot(dx, dy)

                if incremental_distance < 5.0:  # Only accumulate if distance jump is small
                    self._cumulative_moved_distance += incremental_distance

                self._last_stop_location = self._current_position

            moved_distance = self._cumulative_moved_distance
            remaining_distance = max(self._stop_distance - moved_distance, 0.0)

            # ADD THIS OVERSHOOT PROTECTION
            if moved_distance >= self._stop_distance - 6.0:
                throttle = 0.0
                brake = 1.0
            else:
                # Decide throttle or brake normally
                throttle, brake, action = self.stopping_control_logic(self._current_speed / 3.6, remaining_distance)

            # Steering should still use Pure Pursuit even during STOP
            steering = self.lateral_controller.pure_pursuit(
                self._waypoints,
                self._current_position.x,
                self._current_position.y,
                self._current_yaw,
                self._current_speed / 3.6
            )

            return throttle, brake, steering

        else:
            # --- PROCEED/FOLLOW Mode ---
            if self._proceeding_after_stop:
                # Parameters for 15-second ramp-up (very gentle acceleration)
                ramp_duration = 9.0  # seconds to reach target throttle
                target_throttle =  0.2 # Lower target throttle for minimal energy
                
                # Time since PROCEED/FOLLOW started
                elapsed = time.time() - self._proceed_start_time
                x = (elapsed / ramp_duration) * 6.0 - 3.0  # x ∈ [-3, 3]
                sigmoid = 0.5 * (1 + math.tanh(x))         # sigmoid ∈ [0, 1]
                throttle = target_throttle * sigmoid

                if sigmoid >= 0.999:
                    throttle = target_throttle
                    self._throttle_locked_at_max = True
                    self._proceeding_after_stop = False  # disable ramp logic once done

                brake = 0.0

            else:
                # If not in ramp-up phase, maintain constant throttle
                # Use minimal throttle needed to maintain speed
                throttle = 0.2  # Minimal constant throttle
                brake = 0.0

            # Calculate steering using Pure Pursuit
            steering = self.lateral_controller.pure_pursuit(
                self._waypoints,
                self._current_position.x,
                self._current_position.y,
                self._current_yaw,
                self._current_speed / 3.6
            )

            # Compute acceleration from speed change (if needed)
            dv = self._current_speed - self._last_energy_speed
            dt = time.time() - self._last_energy_timestamp
            acceleration = (dv / dt) / 3.6 if dt > 0 else 0.0

            instant_energy_kwh = self.update_energy_usage(acceleration=acceleration)
            self._last_energy_speed = self._current_speed

            return throttle, brake, steering, instant_energy_kwh

    def update_waypoints(self, waypoints):
        self._waypoints = waypoints
        self.longitudinal_controller.update_waypoints(waypoints)

    def is_destination_reached(self):
        total_energy_kWh = self._total_energy_joules / 3600000

        if not self._waypoints:
            return False

        last_waypoint = self._waypoints[-1]
        distance = math.sqrt((self._current_position.x - last_waypoint[0])**2 + 
                           (self._current_position.y - last_waypoint[1])**2)
        return distance < 2.0

    def load_waypoints_from_csv(self, filename):
        try:
            waypoints = []
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # skip the header
                for row in reader:
                    if len(row) >= 2:
                        x = float(row[0])
                        y = float(row[1])
                        waypoints.append((x, y)) 
            if not waypoints:
                return False
            self._waypoints = waypoints
            return True
        except Exception as e:
            return False
