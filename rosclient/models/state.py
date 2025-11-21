"""Connection state enumeration."""
from enum import Enum


class ConnectionState(Enum):
    """Connection state enumeration for ROS clients."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    TIMEOUT = "timeout"
    CLOSED = "closed"
    RECONNECTING = "reconnecting"
    RECONNECTED = "reconnected"

