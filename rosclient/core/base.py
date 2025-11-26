"""Base class for ROS clients."""
from __future__ import annotations

import math
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from ..models.drone import DroneState
from ..models.state import ConnectionState
from ..utils.logger import setup_logger


class RosClientBase(ABC):
    """Base class for managing the lifecycle of a ROS client."""

    def __init__(self, connection_str: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base ROS client.
        
        Args:
            connection_str: Connection string (e.g., WebSocket URL)
            config: Optional configuration dictionary
        """
        self.connection_str = connection_str
        self._config = dict(config or {})
        self._lock = threading.RLock()
        self._state = DroneState()
        self._stop = threading.Event()
        self._connection_state = ConnectionState.DISCONNECTED
        self._connect_lock = threading.Lock()
        self._connecting = False
        self.log = setup_logger(self.__class__.__name__)

    def is_connected(self) -> bool:
        """
        Check if the client is connected.
        
        Returns:
            True if connected, False otherwise
        """
        with self._lock:
            return self._connection_state == ConnectionState.CONNECTED

    @abstractmethod
    def connect_async(self) -> None:
        """Asynchronously connect to ROS. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def terminate(self) -> None:
        """Terminate the connection. Must be implemented by subclasses."""
        pass
    
    def get_status(self) -> DroneState:
        """
        Get the current drone state.
        
        Returns:
            Current drone state
        """
        with self._lock:
            return self._state
    
    def get_position(self) -> tuple[float, float, float]:
        """
        Get the current position as (latitude, longitude, altitude).
        
        Returns:
            Tuple of (latitude, longitude, altitude)
        """
        with self._lock:
            return (self._state.latitude, self._state.longitude, self._state.altitude)
    
    def get_orientation(self) -> tuple[float, float, float]:
        """
        Get the current orientation as (roll, pitch, yaw) in degrees.
        
        Returns:
            Tuple of (roll, pitch, yaw) in degrees
        """
        with self._lock:
            return (self._state.roll, self._state.pitch, self._state.yaw)

    def update_odom(self, msg: Dict[str, Any]) -> None:
        """
        Update odometry from ROS message.
        
        Args:
            msg: Odometry message dictionary
        """
        try:
            with self._lock:
                # Extract quaternion
                q = msg.get("pose", {}).get("pose", {}).get("orientation", {}) or {}
                x = float(q.get("x", 0.0))
                y = float(q.get("y", 0.0))
                z = float(q.get("z", 0.0))
                w = float(q.get("w", 1.0))

                # quaternion -> euler
                sinr_cosp = 2.0 * (w * x + y * z)
                cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
                roll = math.atan2(sinr_cosp, cosr_cosp)

                sinp = 2.0 * (w * y - z * x)
                if abs(sinp) >= 1:
                    pitch = math.copysign(math.pi / 2, sinp)
                else:
                    pitch = math.asin(sinp)

                siny_cosp = 2.0 * (w * z + x * y)
                cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
                yaw = math.atan2(siny_cosp, cosy_cosp)

                roll_deg = math.degrees(roll)
                pitch_deg = math.degrees(pitch)
                yaw_deg = math.degrees(yaw)

                # update state
                self._state.roll = roll_deg
                self._state.pitch = pitch_deg
                self._state.yaw = yaw_deg

                pos = msg.get("pose", {}).get("pose", {}).get("position", {}) or {}
                # use provided position if present (x,y,z often in local odom frame)
                try:
                    self._state.latitude = float(pos.get("x", self._state.latitude))
                    self._state.longitude = float(pos.get("y", self._state.longitude))
                    self._state.altitude = float(pos.get("z", self._state.altitude))
                except Exception:
                    self.log.debug("Partial or invalid odometry position data; skipping position update")

                self._state.last_updated = time.time()
        except Exception as e:
            self.log.error(f"Error handling odometry update: {e}")

