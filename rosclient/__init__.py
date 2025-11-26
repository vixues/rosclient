"""ROS Client library for drone control."""
from .clients import RosClient, MockRosClient
from .models import DroneState, RosTopic, ConnectionState
from .core import RosClientBase, TopicServiceManager

try:
    from .clients import RosbagClient
    HAS_ROSBAG = True
except ImportError:
    HAS_ROSBAG = False
    RosbagClient = None

__version__ = "1.0.0"
__all__ = [
    'RosClient',
    'MockRosClient',
    'DroneState',
    'RosTopic',
    'ConnectionState',
    'RosClientBase',
    'TopicServiceManager',
]
if HAS_ROSBAG:
    __all__.append('RosbagClient')

