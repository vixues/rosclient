"""Base class for ROS clients."""
from __future__ import annotations

import copy
import math
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from ..models.drone import DroneState
from ..models.state import ConnectionState
from ..utils.logger import setup_logger
from .recorder import Recorder


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
        
        # Recording support
        self._recorder: Optional[Recorder] = None
        self._recording_config = self._config.get("recording", {})
        
        # State synchronization support
        self._state_sync_enabled = self._config.get("state_sync_enabled", True)
        self._state_history: list[tuple[DroneState, float]] = []  # (state, timestamp) pairs
        self._max_state_history = self._config.get("max_state_history", 100)

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

                update_timestamp = time.time()
                self._state.last_updated = update_timestamp
                # Add to state history for synchronization
                self._add_state_to_history(self._state, update_timestamp)
            self.log.debug(f"Odometry updated: roll={roll_deg:.3f}, pitch={pitch_deg:.3f}, yaw={yaw_deg:.3f}")
            
            # Record state if recording is enabled
            if self._recorder and self._recorder.is_recording():
                with self._lock:
                    self._recorder.record_state(self._state, update_timestamp)
        except Exception as e:
            self.log.exception(f"Error handling odometry update: {e}")
    
    # ---------- Recording methods ----------
    
    def start_recording(
        self,
        record_images: bool = True,
        record_pointclouds: bool = True,
        record_states: bool = True,
        image_quality: int = 85,
        **kwargs
    ) -> bool:
        """
        Start recording data from the client.
        
        Args:
            record_images: Whether to record images
            record_pointclouds: Whether to record point clouds
            record_states: Whether to record states
            image_quality: JPEG quality (1-100) for image compression
            **kwargs: Additional recorder configuration
            
        Returns:
            True if recording started successfully
        """
        try:
            if self._recorder and self._recorder.is_recording():
                self.log.warning("Recording already in progress")
                return False
            
            # Create recorder if needed
            if self._recorder is None:
                self._recorder = Recorder(
                    record_images=record_images,
                    record_pointclouds=record_pointclouds,
                    record_states=record_states,
                    image_quality=image_quality,
                    max_queue_size=self._recording_config.get("max_queue_size", 100),
                    batch_size=self._recording_config.get("batch_size", 10),
                    logger=self.log
                )
            
            # Start recording
            self._recorder.start_recording(
                client_type=self.__class__.__name__,
                connection_str=self.connection_str,
                config=self._config
            )
            
            self.log.info("Recording started")
            return True
        except Exception as e:
            self.log.error(f"Failed to start recording: {e}")
            return False
    
    def stop_recording(self) -> bool:
        """
        Stop recording.
        
        Returns:
            True if recording stopped successfully
        """
        try:
            if self._recorder and self._recorder.is_recording():
                self._recorder.stop_recording()
                self.log.info("Recording stopped")
                return True
            return False
        except Exception as e:
            self.log.error(f"Failed to stop recording: {e}")
            return False
    
    def save_recording(self, file_path: str, compress: bool = True) -> bool:
        """
        Save recorded data to file.
        
        Args:
            file_path: Path to save file
            compress: Whether to compress the file
            
        Returns:
            True if saved successfully
        """
        try:
            if not self._recorder:
                self.log.error("No recorder available")
                return False
            
            return self._recorder.save(file_path, compress=compress)
        except Exception as e:
            self.log.error(f"Failed to save recording: {e}")
            return False
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recorder is not None and self._recorder.is_recording()
    
    def get_recorder(self) -> Optional[Recorder]:
        """Get the recorder instance."""
        return self._recorder
    
    def get_recording_statistics(self) -> Optional[Dict[str, Any]]:
        """Get recording statistics."""
        if self._recorder:
            return self._recorder.get_statistics()
        return None
    
    # ---------- State synchronization methods ----------
    
    def _add_state_to_history(self, state: DroneState, timestamp: float) -> None:
        """
        Add state snapshot to history for synchronization.
        
        Args:
            state: State snapshot
            timestamp: Timestamp of the state
        """
        if not self._state_sync_enabled:
            return
        
        with self._lock:
            # Create a deep copy to avoid reference issues
            state_copy = copy.deepcopy(state)
            self._state_history.append((state_copy, timestamp))
            
            # Keep history size bounded
            if len(self._state_history) > self._max_state_history:
                self._state_history.pop(0)
    
    def get_state_at_timestamp(self, timestamp: float, tolerance: float = 0.1) -> Optional[DroneState]:
        """
        Get state snapshot closest to the given timestamp.
        
        Args:
            timestamp: Target timestamp
            tolerance: Maximum time difference in seconds
            
        Returns:
            State snapshot or None if not found within tolerance
        """
        if not self._state_sync_enabled or not self._state_history:
            # Fallback to current state
            with self._lock:
                return copy.deepcopy(self._state)
        
        with self._lock:
            # Find closest state in history
            best_state = None
            best_diff = float('inf')
            
            for state, state_ts in self._state_history:
                diff = abs(state_ts - timestamp)
                if diff < best_diff:
                    best_diff = diff
                    best_state = state
            
            # If within tolerance, return it; otherwise return current state
            if best_diff <= tolerance and best_state:
                return copy.deepcopy(best_state)
            else:
                # Return current state as fallback
                return copy.deepcopy(self._state)
    
    def _update_state_with_timestamp(self, timestamp: float) -> None:
        """
        Update state's last_updated timestamp and add to history.
        
        Args:
            timestamp: Timestamp to set
        """
        with self._lock:
            self._state.last_updated = timestamp
            self._add_state_to_history(self._state, timestamp)
    
    def sync_state_with_data(self, data_timestamp: float) -> DroneState:
        """
        Get synchronized state for a given data timestamp.
        This ensures state is aligned with image/pointcloud timestamps.
        
        Args:
            data_timestamp: Timestamp of the data (image/pointcloud)
            
        Returns:
            Synchronized state snapshot
        """
        if not self._state_sync_enabled:
            with self._lock:
                return copy.deepcopy(self._state)
        
        # Get state closest to data timestamp
        synced_state = self.get_state_at_timestamp(data_timestamp, tolerance=0.5)
        if synced_state:
            return synced_state
        
        # Fallback: return current state
        with self._lock:
            return copy.deepcopy(self._state)

