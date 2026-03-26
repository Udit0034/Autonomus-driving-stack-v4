from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    
    # 1. The Bridge (Starts immediately at T=0)
    carla_node = Node(
        package='my_pkg',
        executable='carla_node',
        name='carla_node',
        output='screen'
    )

    # 2. The Brain
    ekf_node = Node(
        package='my_pkg',
        executable='ekf_node',
        name='ekf_node',
        output='screen'
    )

    # 3. The Black Box
    logger_node = Node(
        package='my_pkg',
        executable='logger_node',
        name='logger_node',
        output='screen'
    )

    # 4. The Muscle
    control_node = Node(
        package='my_pkg',
        executable='control_node',
        name='control_node',
        output='screen'
    )

    # --- THE FIX: Individual Staggered Timers ---
    
    # Start EKF at T = 5.0s
    timer_ekf = TimerAction(
        period=5.0,
        actions=[ekf_node]
    )
    
    # Start Logger at T = 5.0s
    timer_logger = TimerAction(
        period=5.0,
        actions=[logger_node]
    )

    # Start Control Node at T = 8.0s (Gives EKF 3 seconds to boot and subscribe)
    timer_control = TimerAction(
        period=8.0,
        actions=[control_node]
    )

    return LaunchDescription([
        carla_node,
        timer_ekf,
        timer_logger,
        timer_control
    ])