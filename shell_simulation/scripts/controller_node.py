#!/usr/bin/env python3
import rospy
from std_msgs.msg import String, Float64, Float32
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Imu
from shell_simulation.controller import VehicleController, ControlMode

class ControllerNode:
    def __init__(self):
        # Initialize node first
        rospy.init_node('controller_node', anonymous=True)
        
        # Initialize controller after ROS node is initialized
        self.controller = VehicleController(self)
        self.vehicle_state_initialized = False
        self.stop_mode_triggered = False
        self._stop_applied = False
        self._stop_threshold_override = 5.0
        
        # Subscribers
        rospy.Subscriber(
            '/planner/mode',
            String,
            self.mode_callback
        )
            
        rospy.Subscriber(
            '/planner/trajectory/global',
            Path,
            self.global_path_callback
        )
            
        rospy.Subscriber(
            '/carla/ego_vehicle/odometry',
            Odometry,
            self.odometry_callback
        )
            
        rospy.Subscriber(
            '/carla/ego_vehicle/imu',
            Imu,
            self.imu_callback
        )
            
        rospy.Subscriber(
            '/carla/ego_vehicle/speedometer',
            Float32,
            self.speedometer_callback
        )

        # Publishers
        self.throttle_pub = rospy.Publisher('/throttle_command', Float64, queue_size=10)
        self.brake_pub = rospy.Publisher('/brake_command', Float64, queue_size=10)
        self.steering_pub = rospy.Publisher('/steering_command', Float64, queue_size=10)
        self.gear_pub = rospy.Publisher('/gear_command', String, queue_size=10)

        # Set gear to forward immediately
        gear_msg = String()
        gear_msg.data = 'forward'
        self.gear_pub.publish(gear_msg)

        # State variables
        self.current_mode = None
        self.odometry_msg = None
        self.imu_msg = None
        self.speedometer_msg = None

        # Control Timer
        self.control_timer = rospy.Timer(rospy.Duration(0.03), self.control_callback)

    
    def parse_mode_string(self, mode_str):
        if ':' in mode_str:
            mode, param = mode_str.split(':')
            return mode, param
        return mode_str, None

    def mode_callback(self, msg):
        mode, param = self.parse_mode_string(msg.data)
        self.current_mode = mode

        if mode == 'follow' and param:
            speed_mps = float(param)
            speed_kmh = speed_mps
            self.controller.set_mode(ControlMode.FOLLOW, target_speed=speed_kmh)
            self.controller.set_speed_override(speed_kmh)

        elif mode == 'stop' and param:
            stop_distance = float(param)
    
            if not self._stop_applied or stop_distance < self._stop_threshold_override:
                self.controller.set_mode(ControlMode.STOP, stop_distance=stop_distance)

                # Set the flag only for the first non-trivial stop
                if stop_distance >= self._stop_threshold_override:
                    self._stop_applied = True
            

        elif mode == 'proceed':
            self.controller.set_mode(ControlMode.PROCEED)
            self._stop_applied = False  # Reset on proceed

        elif mode == 'maneuver':
            self.controller.set_mode(ControlMode.MANEUVER)

    def global_path_callback(self, msg):
        if self.current_mode != 'maneuver':
            waypoints = [(pose.pose.position.x, pose.pose.position.y) 
                        for pose in msg.poses]
            self.controller.update_waypoints(waypoints)

    def odometry_callback(self, msg):
        self.odometry_msg = msg
        self.controller.update_odometry(msg)
        if not self.vehicle_state_initialized:
            self.vehicle_state_initialized = True

    def imu_callback(self, msg):
        self.imu_msg = msg
        self.controller.update_imu(msg)

    def speedometer_callback(self, msg):
        self.speedometer_msg = msg
        self.controller.update_speed(msg.data)

    def control_callback(self, event):
        if not all([self.odometry_msg, self.imu_msg, self.speedometer_msg]):
            return

        if self.controller._current_position is None:
            return

        try:
            outputs = self.controller.step_go()
            if len(outputs) == 4:
                throttle, brake, steering, instant_energy_kwh = outputs
            else:
                throttle, brake, steering = outputs
                instant_energy_kwh = 0.0  # No motion → no energy consumption

            if self.speedometer_msg and self.speedometer_msg.data < 0.5:
                gear_msg = String()
                gear_msg.data = 'forward'
                self.gear_pub.publish(gear_msg)
            
            # Publish control commands
            self.throttle_pub.publish(Float64(data=throttle))
            self.brake_pub.publish(Float64(data=brake))
            self.steering_pub.publish(Float64(data=steering))

            # Optional: throttle energy log printing every 5s
            if not hasattr(self, '_last_energy_log_time'):
                self._last_energy_log_time = rospy.get_time()

            now = rospy.get_time()
            if now - self._last_energy_log_time > 5.0:
                total_energy_kwh = self.controller._total_energy_joules / 3600000
                rospy.loginfo(f"[ENERGY] Total Energy: {total_energy_kwh:.6f} kWh")
                self._last_energy_log_time = now

        except Exception as e:
            rospy.logerr(f'Error in control callback: {str(e)}')

def main():
    try:
        node = ControllerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        try:
            # Sleep briefly to allow final data to flush
            rospy.sleep(1.0)

        except Exception as e:
            rospy.logerr(f"Error in shutdown sequence: {str(e)}")

if __name__ == '__main__':
    main() 
