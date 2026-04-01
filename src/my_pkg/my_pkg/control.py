import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

import carla
import random
import math

from agents.navigation.basic_agent import BasicAgent


def quaternion_to_yaw(q):
    """Helper function to extract the yaw (heading) angle from a ROS 2 Quaternion."""
    x = q.x
    y = q.y
    z = q.z
    w = q.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class PIDController:
    """PID controller with integral anti-windup clamping."""
    def __init__(self, kp: float, ki: float, kd: float,
                 output_min: float = -1.0, output_max: float = 1.0,
                 integral_clamp: float = 2.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_clamp = integral_clamp
        self._integral = 0.0
        self._prev_error = 0.0
        self._first = True

    def compute(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0

        # Integrate and clamp (anti-windup)
        self._integral += error * dt
        self._integral = max(-self.integral_clamp, min(self.integral_clamp, self._integral))

        if self._first:
            derivative = 0.0
            self._first = False
        else:
            derivative = (error - self._prev_error) / dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        
        # Output saturation
        output = max(self.output_min, min(self.output_max, output))
        return output

class LongitudinalController:
    """Longitudinal controller: Speed tracking + Jerk limiter."""
    def __init__(self, kp=0.6, ki=0.2, kd=0.15, max_accel=2.5, max_decel=3.5,
                 max_jerk=1.0, max_jerk_emergency=5.0, velocity_filter_alpha=0.3):
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.max_jerk = max_jerk
        self.max_jerk_emergency = max_jerk_emergency
        self.velocity_filter_alpha = velocity_filter_alpha

        self._prev_accel = 0.0
        self._filtered_speed = 0.0
        self._prev_target_speed = None 
        self._emergency = False

        self._pid = PIDController(kp, ki, kd, output_min=-max_decel, output_max=max_accel)

    def compute(self, current_speed: float, target_speed: float, dt: float) -> float:
        # 1. Velocity low-pass filter (Smooths out sensor noise)
        self._filtered_speed = (self.velocity_filter_alpha * current_speed + 
                                (1.0 - self.velocity_filter_alpha) * self._filtered_speed)

        # 2. Check for emergency stop (target_speed == 0)
        self._emergency = (target_speed < 0.1)

        # 3. Target speed rate limiter (prevents abrupt jumps)
        if self._prev_target_speed is not None:
            rate = 10.0 if self._emergency else 3.0
            max_change = rate * dt
            target_speed = max(self._prev_target_speed - max_change,
                               min(self._prev_target_speed + max_change, target_speed))

        # 4. Feed-forward acceleration
        a_ff = 0.0 if self._prev_target_speed is None else (target_speed - self._prev_target_speed) / dt if dt > 0.0 else 0.0
        a_ff = max(-self.max_decel, min(self.max_accel, a_ff))
        self._prev_target_speed = target_speed

        # 5. PID on speed error
        error = target_speed - self._filtered_speed
        accel_pid = self._pid.compute(error, dt)

        # 6. Combined command
        accel = a_ff + accel_pid
        accel = max(-self.max_decel, min(self.max_accel, accel))

        # 7. Jerk limiter (Comfort constraint)
        if dt > 0.0:
            jerk_limit = self.max_jerk_emergency if self._emergency else self.max_jerk
            max_change = jerk_limit * dt
            accel = max(self._prev_accel - max_change, min(self._prev_accel + max_change, accel))

        self._prev_accel = accel
        return accel

class LateralController:
    """Lateral controller: Heading error tracking (Bicycle Model approximation)."""
    def __init__(self, kp=1.2, ki=0.05, kd=0.1):
        self._pid = PIDController(kp, ki, kd, output_min=-1.0, output_max=1.0)

    def compute(self, ego_transform: carla.Transform, target_waypoint: carla.Waypoint, dt: float) -> float:
        ego_loc = ego_transform.location
        ego_yaw = math.radians(ego_transform.rotation.yaw)
        
        target_loc = target_waypoint.transform.location
        
        # Calculate angle from car to target waypoint
        v_x = target_loc.x - ego_loc.x
        v_y = target_loc.y - ego_loc.y
        target_yaw = math.atan2(v_y, v_x)
        
        # Calculate heading error
        error = target_yaw - ego_yaw
        
        # Normalize error to [-pi, pi] to prevent windup on full circles
        while error > math.pi: error -= 2.0 * math.pi
        while error < -math.pi: error += 2.0 * math.pi

        # Compute PID steering command
        return self._pid.compute(error, dt)

class ControlNode(Node):
    """
    The main controler of the autonomous stack. 
    Subscribes to the EKF /estimated_state to bypass noisy raw sensors.
    Uses CARLA's BasicAgent for high-level routing and hazard detection, 
    and applies custom Longitudinal and Lateral PID control to drive the vehicle.
    """
    def __init__(self):
        super().__init__('control_node')

        # Connect to CARLA
        client = carla.Client('host.docker.internal', 2000)
        client.set_timeout(60.0)
        self.world = client.get_world()

        # Wait for vehicle to be spawned by CarlaNode
        vehicle = None
        max_retries = 20
        for attempt in range(max_retries):
            vehicles = self.world.get_actors().filter('vehicle.*')
            if len(vehicles) > 0:
                vehicle = vehicles[0]
                self.get_logger().info(f'Found vehicle on attempt {attempt + 1}')
                break
            self.get_logger().info(f'Waiting for vehicle to spawn... (attempt {attempt + 1}/{max_retries})')
            import time
            time.sleep(0.5)
        
        if vehicle is None:
            raise RuntimeError('No vehicle found in CARLA world after waiting')
        
        self.vehicle = vehicle

        # Planner Setup
        self.target_speed_kmh = 60.0
        self.agent = BasicAgent(self.vehicle, target_speed=self.target_speed_kmh)

        spawn_points = self.world.get_map().get_spawn_points()
        self.agent.set_destination(random.choice(spawn_points).location)

        # Initialize Controllers
        self.long_controller = LongitudinalController()
        self.lat_controller = LateralController()

        # Subscribe to EKF State
        self.create_subscription(Odometry, '/estimated_state', self.state_callback, 10)

        self.last_time = None
        self.get_logger().info("Control Node Started!")

    def state_callback(self, msg):
        """Main control loop triggered by the EKF state update. 
        Calculates steering, throttle, and braking using waypoint look-ahead 
        and a dynamic cornering speed limiter."""
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.last_time is None:
            self.last_time = t
            return

        dt = t - self.last_time
        self.last_time = t

        # ===== 1. Extract EKF Estimated State =====
        est_x = msg.pose.pose.position.x
        est_y = msg.pose.pose.position.y
        est_v = msg.twist.twist.linear.x
        est_yaw = quaternion_to_yaw(msg.pose.pose.orientation)

        estimated_transform = carla.Transform(
            carla.Location(x=est_x, y=est_y, z=0.0),
            carla.Rotation(pitch=0.0, yaw=math.degrees(est_yaw), roll=0.0)
        )

        # ===== 2. High-Level Planner (BasicAgent Hazard Detection) =====
        if self.agent.done():
            self.get_logger().info("Destination reached! Calculating a new route...")
            spawn_points = self.world.get_map().get_spawn_points()
            self.agent.set_destination(random.choice(spawn_points).location)

        # Let BasicAgent check for traffic lights/stop signs
        agent_control = self.agent.run_step()

        if agent_control.brake > 0.0 or agent_control.hand_brake:
            target_speed_ms = 0.0
        else:
            target_speed_ms = self.target_speed_kmh / 3.6

        # ===== 3. Get Target Waypoint (With Smooth Cornering Fix) =====
        plan = self.agent.get_local_planner().get_plan()
        if len(plan) > 0:
            # Look slightly ahead to trace corners instead of cutting them
            idx = min(2, len(plan) - 1)
            target_wp = plan[idx][0]
        else:
            # Emergency brake if route is lost
            target_speed_ms = 0.0
            target_wp = self.world.get_map().get_waypoint(estimated_transform.location)

        # ===== 4. Run Custom PID Controllers =====
        steer = self.lat_controller.compute(estimated_transform, target_wp, dt)

        # Cornering Speed Limiter: Slow down naturally on sharp turns
        speed_multiplier = max(0.3, 1.0 - abs(steer))
        safe_target_speed = target_speed_ms * speed_multiplier

        accel = self.long_controller.compute(est_v, safe_target_speed, dt)

        # ===== 5. Map Control to CARLA Vehicle =====
        control = carla.VehicleControl()
        control.steer = steer

        if accel >= 0.0:
            control.throttle = min(1.0, accel / self.long_controller.max_accel)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(1.0, abs(accel) / self.long_controller.max_decel)

        self.vehicle.apply_control(control)

def main():
    try:
        rclpy.init()
        node = ControlNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals() and node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()