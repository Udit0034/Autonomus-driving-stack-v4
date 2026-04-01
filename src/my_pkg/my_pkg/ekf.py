import rclpy
from rclpy.node import Node
import numpy as np
import math

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Quaternion


def yaw_from_quaternion(q):
    """Helper function to get yaw from quaternion"""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def quaternion_from_yaw(yaw):
    """Helper function to get quaternion from yaw"""
    q = Quaternion()
    q.w = math.cos(yaw / 2.0)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    return q


class EKF4State:
    """4-State Extended Kalman Filter: [x, y, v, yaw]"""
    IX, IY, IV, IYAW = 0, 1, 2, 3
    STATE_DIM = 4

    def __init__(self):
        
        self.x = np.zeros((self.STATE_DIM, 1))
        # P: State error covariance matrix
        self.P = np.eye(self.STATE_DIM) * 1.0 
        
        # Q: Process noise covariance matrix
        self.Q = np.diag([0.5, 0.5, 0.1, 0.01]) 

        # R: Measurement noise covariance matrix
        self.R_gnss = np.diag([1.0, 1.0]) 
        self.R_odom = np.array([[0.05]])   
        self.R_compass = np.array([[2.0]])
        
        self._initialized = False

    def initialize_state(self, x, y, v, yaw):
        self.x = np.array([[x], [y], [v], [yaw]])
        self._initialized = True

    def predict(self, dt: float, accel: float, yaw_rate: float):
        """Predict step: Updates the state estimate [x, y, v, yaw] and covariance matrix 
        using IMU linear acceleration and gyroscope yaw rate via the bicycle kinematic model."""
        if not self._initialized or dt <= 0.0: return
        x, y, v, yaw = self.x.flatten()

        x_new = x + v * np.cos(yaw) * dt
        y_new = y + v * np.sin(yaw) * dt
        v_new = v + accel * dt
        yaw_new = yaw + yaw_rate * dt

        self.x = np.array([[x_new], [y_new], [v_new], [yaw_new]])

        F = np.eye(self.STATE_DIM)
        F[self.IX, self.IV] = np.cos(yaw) * dt
        F[self.IX, self.IYAW] = -v * np.sin(yaw) * dt
        F[self.IY, self.IV] = np.sin(yaw) * dt
        F[self.IY, self.IYAW] = v * np.cos(yaw) * dt

        self.P = F @ self.P @ F.T + self.Q

    def _update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray, is_angle: bool = False):
        """Update the covariance matrix using Joseph form"""
        y = z - H @ self.x  
        
        # ANGLE WRAPPING
        if is_angle:
            y[0, 0] = (y[0, 0] + np.pi) % (2.0 * np.pi) - np.pi

        S = H @ self.P @ H.T + R  
        K = self.P @ H.T @ np.linalg.inv(S)  
        self.x = self.x + K @ y
        
        # Stable Joseph form covariance update
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def update_gnss(self, meas_x: float, meas_y: float):
        """Measurement update step fusing 1Hz global position data. """
        if not self._initialized: return
        z = np.array([[meas_x], [meas_y]])
        H = np.zeros((2, self.STATE_DIM))
        H[0, self.IX] = 1.0
        H[1, self.IY] = 1.0
        self._update(z, H, self.R_gnss)

    def update_odom(self, meas_v: float):
        """Measurement update step fusing 20Hz noisy vehicle velocity. """
        if not self._initialized: return
        z = np.array([[meas_v]])
        H = np.zeros((1, self.STATE_DIM))
        H[0, self.IV] = 1.0
        self._update(z, H, self.R_odom)

    def update_compass(self, meas_yaw: float):
        """Measurement update step fusing 100Hz aligning magnetometer heading """
        if not self._initialized: return
        z = np.array([[meas_yaw]])
        H = np.zeros((1, self.STATE_DIM))
        H[0, self.IYAW] = 1.0
        self._update(z, H, self.R_compass, is_angle=True)


class EKFNode(Node):
    """ROS2 Node to run EKF4State"""
    def __init__(self):
        super().__init__('ekf_node')
        self.ekf = EKF4State()

        self.last_time = None
        self.compass_offset = None
        
        # Buffers for async sensor data
        self.latest_gnss_x = None
        self.latest_gnss_y = None
        self.latest_odom_v = None
        self.latest_odom_yaw = None

        # Subscribers
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(PoseStamped, '/gnss_local', self.gnss_callback, 10)
        self.create_subscription(Odometry, '/odom_gt', self.odom_callback, 10)

        # Publisher
        self.state_pub = self.create_publisher(Odometry, '/estimated_state', 10)
        self.get_logger().info("EKF Node Started!")

    def odom_callback(self, msg):
        """Get latest odometry data and update if EKF is online"""
        self.latest_odom_v = msg.twist.twist.linear.x
        self.latest_odom_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        
        # Process odometry update if EKF is online {Outside main loop due to high update freq compare to gnss data}
        if self.ekf._initialized:
            self.ekf.update_odom(self.latest_odom_v)

    def gnss_callback(self, msg):
        """Get latest gnss data"""
        self.latest_gnss_x = msg.pose.position.x
        self.latest_gnss_y = msg.pose.position.y

    def imu_callback(self, msg):
        """Main 100Hz loop. Calculates dt, runs the EKF prediction using IMU data, 
        applies the compass measurement update, and safely consumes buffered GNSS data."""
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.last_time is None:
            self.last_time = t
            return

        dt = t - self.last_time
        self.last_time = t

        accel = msg.linear_acceleration.x
        yaw_rate = msg.angular_velocity.z
        compass_yaw = yaw_from_quaternion(msg.orientation)

        # 1. INITIALIZE ON FIRST GNSS & ODOM
        if not self.ekf._initialized:
            if self.latest_gnss_x is not None and self.latest_odom_yaw is not None and self.latest_odom_v is not None:
                self.compass_offset = compass_yaw - self.latest_odom_yaw
                self.ekf.initialize_state(self.latest_gnss_x, self.latest_gnss_y, self.latest_odom_v, self.latest_odom_yaw)
                self.get_logger().info("EKF Initialized with Correct Yaw!")
            return

        # 2. PREDICT
        self.ekf.predict(dt, accel, yaw_rate)

        # 3. UPDATE COMPASS 
        if self.compass_offset is not None:
            aligned_compass = compass_yaw - self.compass_offset
            self.ekf.update_compass(aligned_compass)

        # 4. UPDATE GNSS (If new data arrived)
        if self.latest_gnss_x is not None:
            self.ekf.update_gnss(self.latest_gnss_x, self.latest_gnss_y)
            self.latest_gnss_x = None  # Clear buffer after using

        # 5. PUBLISH ESTIMATED STATE
        self.publish_state(msg.header.stamp)

    def publish_state(self, stamp):
        """Publish the estimated state to /estimated_state topic"""
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'

        st = self.ekf.x.flatten()
        msg.pose.pose.position.x = float(st[0])
        msg.pose.pose.position.y = float(st[1])
        msg.pose.pose.orientation = quaternion_from_yaw(float(st[3]))
        msg.twist.twist.linear.x = float(st[2])

        self.state_pub.publish(msg)

def main():
    """Main function to run node properly and exit without error"""
    rclpy.init()
    node = EKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()