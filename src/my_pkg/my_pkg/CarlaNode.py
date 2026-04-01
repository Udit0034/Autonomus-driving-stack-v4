import rclpy
from rclpy.executors import ShutdownException
import carla
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Odometry
import math
import random

class CarlaNode(Node):
    """CarlaNode class inherited from rclpy.node.Node this class act as the immediate connection between ros2 and carla simulator.
    This class initialize the world, ego_vehicle, sensors.
    Publish sensor data to respective topics ."""
    def __init__(self) -> None:
        super().__init__('carla_node')
        self.actors = self.setup_environment()
        self.world = self.actors['world']
        self.ego_vehicle = self.actors['ego_vehicle']
        self.imu_sensor = self.actors['imu_sensor']
        self.gnss_sensor = self.actors['gnss_sensor']
        self.spectator = self.actors['spectator']
        
        self.imu_pub = self.create_publisher(Imu, '/imu', 10) #Publish imu sensor data to /imu topic
        self.gnss_pub = self.create_publisher(PoseStamped, '/gnss_local', 10) #Publish gnss sensor data to /gnss_)local topic
        self.odom_pub = self.create_publisher(Odometry, '/odom_gt', 10) #Publish odometry ground truth to /odom_gt topic

        self.gnss_sensor.listen(self.gnss_callback)
        self.imu_sensor.listen(self.imu_callback)

        # Publish odometry at 20 Hz to match the physics engine
        self.odom_timer = self.create_timer(0.05, self.publish_odom)
    
    def imu_callback(self, data):
        """Callback to publish imu data to /imu topic by reading linear acceleration and angular velocity on 3 axis.
        and convert euler's angle to Quaternions."""
        msg = Imu()
        msg.linear_acceleration.x = data.accelerometer.x
        msg.linear_acceleration.y = data.accelerometer.y
        msg.linear_acceleration.z = data.accelerometer.z
        
        msg.angular_velocity.x = data.gyroscope.x
        msg.angular_velocity.y = data.gyroscope.y
        msg.angular_velocity.z = data.gyroscope.z

        # Use CARLA's noisy compass for the Yaw instead of Ground Truth
        # We are running a 2D (Planar) EKF, so we assume the car is flat.
        # We zero out pitch and roll to prevent 3D tilt noise from confusing the filter.
        yaw = data.compass 
        pitch = 0.0 
        roll = 0.0
        
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        q = Quaternion()
        q.w = cr * cp * cy + sr * sp * sy
        q.x = sr * cp * cy - cr * sp * sy
        q.y = cr * sp * cy + sr * cp * sy
        q.z = cr * cp * sy - sr * sp * cy
        msg.orientation = q
        
        msg.header.stamp = self.get_clock().now().to_msg()
        self.imu_pub.publish(msg)

    def gnss_callback(self, data):
        """Callback to publish gnss data to /gnss_local topic """
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.position.x = data.transform.location.x
        msg.pose.position.y = data.transform.location.y
        msg.pose.position.z = data.transform.location.z if hasattr(data.transform.location, 'z') else 0.0 # Checking if data.transform.location has a z attribute to safeguard against AttributeError

        self.gnss_pub.publish(msg)
    
    def publish_odom(self):
        """
        Updates the spectator camera to follow the ego vehicle and publishes 
        simulated wheel odometry. Injects Gaussian noise and a 2% scale error 
        into the velocity to simulate real-world hardware imperfections for the EKF.
        """
        if self.ego_vehicle is None:
            return

        gt_transform = self.ego_vehicle.get_transform()
        velocity_vec = self.ego_vehicle.get_velocity()
        
        # Update Spectator
        forward = gt_transform.get_forward_vector()
        self.spectator.set_transform(carla.Transform(
            gt_transform.location - carla.Location(x=forward.x * 6, y=forward.y * 6, z=0) + carla.Location(z=3),
            carla.Rotation(pitch=-15, yaw=gt_transform.rotation.yaw, roll=0)
        ))

        # Generate Odometry data with gaussian noise and drift to prevent ground truth leaking directly and fuse into kalman filter later 
        gt_speed = math.sqrt(velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2)
        odom_speed = gt_speed * 1.02 + random.gauss(0.0, 0.1)
        
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        # Position (Ground Truth for PID target calculation reference)
        odom_msg.pose.pose.position.x = gt_transform.location.x
        odom_msg.pose.pose.position.y = gt_transform.location.y
        odom_msg.pose.pose.position.z = gt_transform.location.z

        # Orientation (Ground Truth Yaw for initialization and PID tracking)
        yaw_rad = math.radians(gt_transform.rotation.yaw)
        odom_msg.pose.pose.orientation.w = math.cos(yaw_rad / 2.0)
        odom_msg.pose.pose.orientation.x = 0.0
        odom_msg.pose.pose.orientation.y = 0.0
        odom_msg.pose.pose.orientation.z = math.sin(yaw_rad / 2.0)

        # Velocity (NOISY Odometry for EKF)
        odom_msg.twist.twist.linear.x = odom_speed
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.linear.z = 0.0

        self.odom_pub.publish(odom_msg)

    def setup_environment(self):
        """Initial function to be called in constructor it  connect to carla simulator on host.docker.internal:2000 and load map 'Town1' .
        Spawning a Tesla Model 3 as ego vehicle and configuring sensors gnss ,  imu and spectator """
        client = carla.Client('host.docker.internal', 2000)
        client.set_timeout(60.0)
        world = client.load_world('Town01')
        
        # Lock physics to 20 FPS to sync with ROS 2
        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        blueprint_library = world.get_blueprint_library()

        tesla_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[0]
        ego_vehicle = world.spawn_actor(tesla_bp, spawn_point)

        imu_bp = blueprint_library.filter('sensor.other.imu')[0]
        gnss_bp = blueprint_library.filter('sensor.other.gnss')[0]

        # imu publish data at 100Hz while gnss on 1Hz to simulate realistic asynchronous hardware rates.
        imu_bp.set_attribute('sensor_tick', '0.01')
        gnss_bp.set_attribute('sensor_tick', '1.0')

        sensor_transform = carla.Transform(carla.Location(x=0, z=0)) # placeing sensors at centor of ego_vehicle

        imu_sensor = world.spawn_actor(imu_bp, sensor_transform, attach_to=ego_vehicle)
        gnss_sensor = world.spawn_actor(gnss_bp, sensor_transform, attach_to=ego_vehicle)

        spectator = world.get_spectator()

        return {
            'world': world,
            'ego_vehicle': ego_vehicle,
            'imu_sensor': imu_sensor,
            'gnss_sensor': gnss_sensor,
            'spectator': spectator,
        }

def main():
    """Main function to run node properly and exit without error"""
    node = None
    try:
        rclpy.init()
        node = CarlaNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ShutdownException):
        pass
    finally:
        if node is not None:
            # Clean up actor
            node.get_logger().info("Cleaning up CARLA actors")
            node.actors['imu_sensor'].stop()
            node.actors['gnss_sensor'].stop()
            node.actors['imu_sensor'].destroy()
            node.actors['gnss_sensor'].destroy()
            node.actors['ego_vehicle'].destroy()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()