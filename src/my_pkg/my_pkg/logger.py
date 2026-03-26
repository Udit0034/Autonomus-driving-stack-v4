import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import csv
import os
import math

RESULTS_DIR = '/workspace/results'

# --- Helper Function to get Yaw ---
def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

class LoggerNode(Node):
    def __init__(self):
        super().__init__('logger_node')

        # --- Sequential Run Directory Logic ---
        os.makedirs(RESULTS_DIR, exist_ok=True)
        existing_runs = [d for d in os.listdir(RESULTS_DIR) if d.startswith('run_')]
        run_nums = []
        for d in existing_runs:
            parts = d.split('_')
            if len(parts) == 2 and parts[1].isdigit():
                run_nums.append(int(parts[1]))
                
        next_run = max(run_nums) + 1 if run_nums else 1
        self.run_dir = os.path.join(RESULTS_DIR, f"run_{next_run}")
        os.makedirs(self.run_dir, exist_ok=True)

        # --- CSV Files ---
        self.odom_file = open(f"{self.run_dir}/odom_data.csv", "w", newline="")
        self.ekf_file = open(f"{self.run_dir}/ekf_data.csv", "w", newline="")

        self.odom_writer = csv.writer(self.odom_file)
        self.ekf_writer = csv.writer(self.ekf_file)

        # Headers mapped for the evaluation script
        self.odom_writer.writerow(["Timestamp", "Loc_X", "Loc_Y", "GT_Velocity", "Yaw_Degrees"])
        self.ekf_writer.writerow(["Timestamp", "Est_X", "Est_Y", "Est_Velocity", "Est_Yaw"])

        # Subscribers
        self.create_subscription(Odometry, '/odom_gt', self.odom_callback, 10)
        self.create_subscription(Odometry, '/estimated_state', self.ekf_callback, 10)

        self.get_logger().info(f"Logging telemetry to {self.run_dir} 📊")

    def get_time(self, msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def odom_callback(self, msg):
        t = self.get_time(msg)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        v = msg.twist.twist.linear.x
        yaw = math.degrees(quaternion_to_yaw(msg.pose.pose.orientation))

        self.odom_writer.writerow([t, x, y, v, yaw])

    def ekf_callback(self, msg):
        t = self.get_time(msg)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        v = msg.twist.twist.linear.x
        yaw = math.degrees(quaternion_to_yaw(msg.pose.pose.orientation))

        self.ekf_writer.writerow([t, x, y, v, yaw])

def main():
    try:
        rclpy.init()
        node = LoggerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals() and node is not None:
            # Safely close files before destroying node
            node.odom_file.close()
            node.ekf_file.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()