"""Base class for ROS clients."""
from __future__ import annotations

import base64
import math
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List

import numpy as np

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

    def _decode_point_cloud(self, msg: Dict[str, Any]) -> Optional[Tuple[np.ndarray, float]]:
        """
        Common decoder for PointCloud2-like messages. Returns (points, timestamp) or None.
        
        Args:
            msg: PointCloud2 message dictionary
            
        Returns:
            Tuple of (points array, timestamp) or None if decoding fails
        """
        try:
            if "data" not in msg or "fields" not in msg:
                return None

            raw = msg["data"]
            # some transports provide base64 strings; roslibpy may provide bytes
            if isinstance(raw, str):
                raw_data = base64.b64decode(raw)
            elif isinstance(raw, (bytes, bytearray)):
                raw_data = bytes(raw)
            else:
                self.log.debug("Unsupported point cloud data type")
                return None

            np_data = np.frombuffer(raw_data, dtype=np.uint8)
            fields = msg["fields"]
            point_step = int(msg.get("point_step", 0))
            if point_step <= 0:
                self.log.debug("Invalid point_step")
                return None

            # find offsets
            x_offset = next((f["offset"] for f in fields if f["name"] == "x"), None)
            y_offset = next((f["offset"] for f in fields if f["name"] == "y"), None)
            z_offset = next((f["offset"] for f in fields if f["name"] == "z"), None)
            if None in (x_offset, y_offset, z_offset):
                self.log.debug("Missing x/y/z fields in PointCloud2.")
                return None

            points: List[Tuple[float, float, float]] = []
            total_len = len(np_data)
            # iterate safely
            for i in range(0, total_len - point_step + 1, point_step):
                # extract 4-byte floats; ensure slice is within bounds
                try:
                    x = np.frombuffer(np_data[i + x_offset:i + x_offset + 4], dtype=np.float32)[0]
                    y = np.frombuffer(np_data[i + y_offset:i + y_offset + 4], dtype=np.float32)[0]
                    z = np.frombuffer(np_data[i + z_offset:i + z_offset + 4], dtype=np.float32)[0]
                    points.append((x, y, z))
                except Exception:
                    # skip malformed point
                    continue

            if not points:
                return None

            points_arr = np.array(points, dtype=np.float32)
            return points_arr, time.time()
        except Exception:
            return None
        
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
            self.log.debug(f"Odometry updated: roll={roll_deg:.3f}, pitch={pitch_deg:.3f}, yaw={yaw_deg:.3f}")
        except Exception as e:
            self.log.exception(f"Error handling odometry update: {e}")

