"""ROS client implementations."""
from .ros_client import RosClient
from .mock_client import MockRosClient

try:
    from .rosbag_client import RosbagClient
    HAS_ROSBAG = True
except ImportError:
    HAS_ROSBAG = False
    RosbagClient = None

try:
    from .airsim_client import AirSimClient
    HAS_AIRSIM = True
except ImportError:
    HAS_AIRSIM = False
    AirSimClient = None

__all__ = ['RosClient', 'MockRosClient']
if HAS_ROSBAG:
    __all__.append('RosbagClient')
if HAS_AIRSIM:
    __all__.append('AirSimClient')

