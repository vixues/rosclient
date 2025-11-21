"""Mock ROS client for testing."""
import logging
import time
from typing import Optional, Dict, Any, List

from ..core.base import RosClientBase
from ..models.state import ConnectionState
from ..utils.logger import setup_logger


class MockRosClient(RosClientBase):
    """Mock ROS client for testing without actual ROS connection."""

    def __init__(self, connection_str: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the mock ROS client.
        
        Args:
            connection_str: Connection string (for compatibility)
            config: Optional configuration dictionary
        """
        super().__init__(connection_str, config=config)
        self._config.setdefault("service_call_timeout", 5.0)
        self._terminated = False
        self.published_messages: List[Dict[str, Any]] = []
        self.service_calls: List[Dict[str, Any]] = []
        
        # initialize a plausible default state
        with self._lock:
            self._state.connected = True
            self._state.armed = False
            self._state.mode = "STANDBY"
            self._state.battery = 100.0
            self._state.latitude = 22.5329
            self._state.longitude = 113.93029
            self._state.altitude = 0.0
            self._connection_state = ConnectionState.CONNECTED

        self.log = setup_logger(f"MockRosClient[{connection_str}]")
        self.log.setLevel(logging.DEBUG)

    def is_connected(self) -> bool:
        """
        Check if the mock client is connected.
        
        Returns:
            True if not terminated and connected
        """
        with self._lock:
            return not getattr(self, "_terminated", False) and bool(self._state.connected)

    def connect_async(self) -> None:
        """Immediately connect in mock mode."""
        with self._lock:
            self._terminated = False
            self._state.connected = True
            self._connection_state = ConnectionState.CONNECTED
            self.log.debug("Mock: connected (connect_async)")

    def terminate(self) -> None:
        """Terminate the mock connection."""
        with self._lock:
            self._terminated = True
            self._state.connected = False
            self._connection_state = ConnectionState.DISCONNECTED
            self.log.debug("Mock: terminated")

    def service_call(self, service_name: str, service_type: str, payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Mock service call.
        
        Args:
            service_name: Service name
            service_type: Service type
            payload: Service request payload
            **kwargs: Additional arguments
            
        Returns:
            Mock service response
        """
        with self._lock:
            self.service_calls.append({
                "service_name": service_name,
                "service_type": service_type,
                "payload": payload
            })
            self.log.debug(f"Mock service call recorded: {service_name}")
            # Simulate a small delay
            time.sleep(0.01)
            return {"mock": "success", "service": service_name}

    def publish(self, topic_name: str, topic_type: str, message: Dict[str, Any], **kwargs) -> None:
        """
        Mock publish.
        
        Args:
            topic_name: Topic name
            topic_type: Topic type
            message: Message dictionary
            **kwargs: Additional arguments
        """
        with self._lock:
            self.published_messages.append({
                "topic_name": topic_name,
                "topic_type": topic_type,
                "message": message
            })
            self.log.debug(f"Mock publish recorded: {topic_name}")

    # state mutation helpers for tests
    def set_mode(self, mode: str) -> None:
        """
        Set the drone mode (for testing).
        
        Args:
            mode: Mode string
        """
        with self._lock:
            self._state.mode = mode
            self._state.last_updated = time.time()

    def set_armed(self, armed: bool) -> None:
        """
        Set the armed state (for testing).
        
        Args:
            armed: Armed state
        """
        with self._lock:
            self._state.armed = armed
            self._state.last_updated = time.time()

    def set_battery(self, percent: float) -> None:
        """
        Set the battery percentage (for testing).
        
        Args:
            percent: Battery percentage
        """
        with self._lock:
            self._state.battery = percent
            self._state.last_updated = time.time()

    def set_position(self, lat: float, lon: float, alt: float) -> None:
        """
        Set the position (for testing).
        
        Args:
            lat: Latitude
            lon: Longitude
            alt: Altitude
        """
        with self._lock:
            self._state.latitude = lat
            self._state.longitude = lon
            self._state.altitude = alt
            self._state.last_updated = time.time()

