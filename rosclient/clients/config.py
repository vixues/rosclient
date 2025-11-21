"""Default configuration for ROS clients."""
from ..models.drone import RosTopic

# Default ROS topics configuration
DEFAULT_TOPICS = {
    "state": RosTopic(name="/mavros/state", type="mavros_msgs/State"),
    "odom": RosTopic(name="/mavros/local_position/odom", type="nav_msgs/Odometry"),
    "battery": RosTopic(name="/mavros/battery", type="sensor_msgs/BatteryState"),
    "gps": RosTopic(name="/mavros/global_position/global", type="sensor_msgs/NavSatFix"),
    "control": RosTopic(name="/control", type="controller_msgs/cmd"),
    "drone_state": RosTopic(name="/mavros/drone_state", type="controller_msgs/DroneState"),
    "goal": RosTopic(name="/goal_user2brig", type="quadrotor_msgs/GoalSet"),
    "camera": RosTopic(name="/camera/image_raw/compressed", type="sensor_msgs/CompressedImage"),
    "point_cloud": RosTopic(name="/drone_1_cloud_registered", type="sensor_msgs/PointCloud2"),
}

# Default configuration
DEFAULT_CONFIG = {
    "connect_max_retries": 5,
    "connect_backoff_seconds": 1.0,
    "connect_backoff_max": 30.0,
    "service_call_timeout": 5.0,
    "service_call_retries": 2,
    "publish_retries": 2,
    "logger_level": 20,  # logging.INFO
}

