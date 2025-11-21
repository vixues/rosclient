"""Mock ROS client for testing."""
import logging
import random
import threading
import time
from typing import Optional, Dict, Any, List, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None

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
        
        # Initialize mock image and point cloud
        self._latest_image: Optional[Tuple] = None
        self._latest_point_cloud: Optional[Tuple] = None
        self._image_update_thread: Optional[threading.Thread] = None
        self._pc_update_thread: Optional[threading.Thread] = None
        self._stop_updates = threading.Event()
        
        # Start background threads to generate random data
        self._start_mock_data_generation()

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
        self._stop_updates.set()
        with self._lock:
            self._terminated = True
            self._state.connected = False
            self._connection_state = ConnectionState.DISCONNECTED
        self.log.debug("Mock: terminated")
        
    def _start_mock_data_generation(self):
        """Start background threads to generate random image and point cloud data."""
        def generate_image():
            while not self._stop_updates.is_set():
                if HAS_CV2 and HAS_NUMPY:
                    # Generate random image (640x480 RGB)
                    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    # Add some patterns to make it more interesting
                    cv2.circle(img, (320, 240), 100, (255, 255, 255), -1)
                    cv2.putText(img, "MOCK", (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
                    timestamp = time.time()
                    with self._lock:
                        self._latest_image = (img, timestamp)
                time.sleep(0.5)  # Update every 0.5 seconds
                
        def generate_pointcloud():
            while not self._stop_updates.is_set():
                if HAS_NUMPY:
                    # Generate random point cloud (1000-5000 points)
                    num_points = random.randint(1000, 5000)
                    # Generate points in a sphere-like shape
                    theta = np.random.uniform(0, 2 * np.pi, num_points)
                    phi = np.random.uniform(0, np.pi, num_points)
                    r = np.random.uniform(1, 5, num_points)
                    x = r * np.sin(phi) * np.cos(theta)
                    y = r * np.sin(phi) * np.sin(theta)
                    z = r * np.cos(phi)
                    points = np.column_stack([x, y, z])
                    timestamp = time.time()
                    with self._lock:
                        self._latest_point_cloud = (points, timestamp)
                time.sleep(1.0)  # Update every second
                
        if HAS_CV2 and HAS_NUMPY:
            self._image_update_thread = threading.Thread(target=generate_image, daemon=True)
            self._image_update_thread.start()
            
        if HAS_NUMPY:
            self._pc_update_thread = threading.Thread(target=generate_pointcloud, daemon=True)
            self._pc_update_thread.start()

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
            
    def get_latest_image(self) -> Optional[Tuple]:
        """
        Get the latest mock image.
        
        Returns:
            Tuple of (image array, timestamp) or None
        """
        with self._lock:
            return getattr(self, "_latest_image", None)
            
    def get_latest_point_cloud(self) -> Optional[Tuple]:
        """
        Get the latest mock point cloud.
        
        Returns:
            Tuple of (points array, timestamp) or None
        """
        with self._lock:
            return getattr(self, "_latest_point_cloud", None)
            
    def fetch_camera_image(self) -> Optional[Tuple]:
        """
        Fetch mock camera image.
        
        Returns:
            Tuple of (image array, timestamp) or None
        """
        return self.get_latest_image()
        
    def fetch_point_cloud(self) -> Optional[Tuple]:
        """
        Fetch mock point cloud.
        
        Returns:
            Tuple of (points array, timestamp) or None
        """
        return self.get_latest_point_cloud()

