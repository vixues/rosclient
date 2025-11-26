"""Mock ROS client for testing."""
import logging
import queue
import random
import threading
import time
from pathlib import Path
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
            config: Optional configuration dictionary. Supported options:
                - real_image_path: Path to a single image file or directory containing images
                                  (supports .jpg, .jpeg, .png, .bmp, .tiff, .tif)
                - real_pointcloud_path: Path to a single point cloud file or directory containing point clouds
                                       (supports .npy, .ply, .pcd)
                - image_update_interval: Update interval for images in seconds (default: 0.1)
                - pointcloud_update_interval: Update interval for point clouds in seconds (default: 0.1)
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
        
        # High-frequency cache for images and point clouds
        # Use queues with maxsize to keep only the latest frames (drop old ones)
        self._image_cache: queue.Queue = queue.Queue(maxsize=5)  # Keep latest 5 frames
        self._pointcloud_cache: queue.Queue = queue.Queue(maxsize=5)  # Keep latest 5 frames
        
        # Configuration for real data sources
        self._real_image_path: Optional[str] = self._config.get("real_image_path")
        self._real_pointcloud_path: Optional[str] = self._config.get("real_pointcloud_path")
        # Reduced intervals for higher refresh rate (default: 0.033s = ~30 FPS)
        self._image_update_interval: float = self._config.get("image_update_interval", 0.033)
        self._pc_update_interval: float = self._config.get("pointcloud_update_interval", 0.033)
        
        # For cycling through multiple files
        self._image_files: List[str] = []
        self._pc_files: List[str] = []
        self._current_image_idx: int = 0
        self._current_pc_idx: int = 0
        
        # Load real data if specified
        if self._real_image_path:
            self._load_image_files()
        if self._real_pointcloud_path:
            self._load_pointcloud_files()
        
        # Start background threads to generate random data or load real data
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
        
    def _load_image_files(self):
        """Load image files from the specified path."""
        if not self._real_image_path:
            return
            
        path = Path(self._real_image_path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        if path.is_file():
            if path.suffix.lower() in image_extensions:
                self._image_files = [str(path)]
                self.log.info(f"Loaded single image file: {path}")
            else:
                self.log.warning(f"Unsupported image format: {path.suffix}")
        elif path.is_dir():
            # Load all image files from directory
            self._image_files = [
                str(p) for p in path.iterdir()
                if p.is_file() and p.suffix.lower() in image_extensions
            ]
            self._image_files.sort()
            self.log.info(f"Loaded {len(self._image_files)} image files from directory: {path}")
        else:
            self.log.warning(f"Image path does not exist: {path}")
            
    def _load_pointcloud_files(self):
        """Load point cloud files from the specified path."""
        if not self._real_pointcloud_path:
            return
            
        path = Path(self._real_pointcloud_path)
        pc_extensions = {'.npy', '.ply', '.pcd'}
        
        if path.is_file():
            if path.suffix.lower() in pc_extensions:
                self._pc_files = [str(path)]
                self.log.info(f"Loaded single point cloud file: {path}")
            else:
                self.log.warning(f"Unsupported point cloud format: {path.suffix}")
        elif path.is_dir():
            # Load all point cloud files from directory
            self._pc_files = [
                str(p) for p in path.iterdir()
                if p.is_file() and p.suffix.lower() in pc_extensions
            ]
            self._pc_files.sort()
            self.log.info(f"Loaded {len(self._pc_files)} point cloud files from directory: {path}")
        else:
            self.log.warning(f"Point cloud path does not exist: {path}")
            
    def _load_image_file(self, file_path: str) -> Optional[np.ndarray]:
        """Load an image file."""
        if not HAS_CV2 or not HAS_NUMPY:
            return None
        try:
            img = cv2.imread(file_path)
            if img is None:
                self.log.warning(f"Failed to load image: {file_path}")
                return None
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            self.log.error(f"Error loading image {file_path}: {e}")
            return None
            
    def _load_pointcloud_file(self, file_path: str) -> Optional[np.ndarray]:
        """Load a point cloud file."""
        if not HAS_NUMPY:
            return None
        try:
            path = Path(file_path)
            suffix = path.suffix.lower()
            
            if suffix == '.npy':
                points = np.load(file_path)
                # Ensure it's a 2D array with 3 columns (x, y, z)
                if points.ndim == 1:
                    return None
                if points.ndim == 2 and points.shape[1] >= 3:
                    return points[:, :3]  # Take only first 3 columns
                return None
            elif suffix == '.ply':
                # Simple PLY loader (ASCII format)
                return self._load_ply_file(file_path)
            elif suffix == '.pcd':
                # Simple PCD loader (ASCII format)
                return self._load_pcd_file(file_path)
            else:
                self.log.warning(f"Unsupported point cloud format: {suffix}")
                return None
        except Exception as e:
            self.log.error(f"Error loading point cloud {file_path}: {e}")
            return None
            
    def _load_ply_file(self, file_path: str) -> Optional[np.ndarray]:
        """Load a PLY file (ASCII format)."""
        try:
            points = []
            with open(file_path, 'r') as f:
                header_ended = False
                for line in f:
                    line = line.strip()
                    if line == "end_header":
                        header_ended = True
                        continue
                    if not header_ended:
                        continue
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            points.append([x, y, z])
                        except ValueError:
                            continue
            if points:
                return np.array(points, dtype=np.float32)
            return None
        except Exception as e:
            self.log.error(f"Error loading PLY file {file_path}: {e}")
            return None
            
    def _load_pcd_file(self, file_path: str) -> Optional[np.ndarray]:
        """Load a PCD file (ASCII format)."""
        try:
            points = []
            with open(file_path, 'r') as f:
                header_ended = False
                for line in f:
                    line = line.strip()
                    if line.startswith("DATA"):
                        if "ascii" in line.lower():
                            header_ended = True
                        continue
                    if not header_ended:
                        continue
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            points.append([x, y, z])
                        except ValueError:
                            continue
            if points:
                return np.array(points, dtype=np.float32)
            return None
        except Exception as e:
            self.log.error(f"Error loading PCD file {file_path}: {e}")
            return None
    
    def _start_mock_data_generation(self):
        """Start background threads to generate random image and point cloud data or load real data."""
        def generate_image():
            while not self._stop_updates.is_set():
                if HAS_CV2 and HAS_NUMPY:
                    img = None
                    
                    # Try to load real image if available
                    if self._image_files:
                        file_path = self._image_files[self._current_image_idx]
                        img = self._load_image_file(file_path)
                        if img is not None:
                            # Cycle to next image
                            self._current_image_idx = (self._current_image_idx + 1) % len(self._image_files)
                    
                    # Fall back to random generation if no real image loaded
                    if img is None:
                        # Generate random image (640x480 RGB)
                        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                        # Add some patterns to make it more interesting
                        cv2.circle(img, (320, 240), 100, (255, 255, 255), -1)
                        cv2.putText(img, "MOCK", (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
                    
                    timestamp = time.time()
                    
                    # Update cache (non-blocking, drop old frames if queue is full)
                    try:
                        self._image_cache.put_nowait((img, timestamp))
                    except queue.Full:
                        # Queue is full, remove oldest and add new
                        try:
                            self._image_cache.get_nowait()
                            self._image_cache.put_nowait((img, timestamp))
                        except queue.Empty:
                            pass
                    
                    # Update legacy latest for backward compatibility
                    with self._lock:
                        self._latest_image = (img, timestamp)
                        
                time.sleep(self._image_update_interval)
                
        def generate_pointcloud():
            while not self._stop_updates.is_set():
                if HAS_NUMPY:
                    points = None
                    
                    # Try to load real point cloud if available
                    if self._pc_files:
                        file_path = self._pc_files[self._current_pc_idx]
                        points = self._load_pointcloud_file(file_path)
                        if points is not None:
                            # Cycle to next point cloud
                            self._current_pc_idx = (self._current_pc_idx + 1) % len(self._pc_files)
                    
                    # Fall back to random generation if no real point cloud loaded
                    if points is None:
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
                    
                    # Update cache (non-blocking, drop old frames if queue is full)
                    try:
                        self._pointcloud_cache.put_nowait((points, timestamp))
                    except queue.Full:
                        # Queue is full, remove oldest and add new
                        try:
                            self._pointcloud_cache.get_nowait()
                            self._pointcloud_cache.put_nowait((points, timestamp))
                        except queue.Empty:
                            pass
                    
                    # Update legacy latest for backward compatibility
                    with self._lock:
                        self._latest_point_cloud = (points, timestamp)
                        
                time.sleep(self._pc_update_interval)
                
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
        Get the latest mock image from cache (non-blocking).
        
        Returns:
            Tuple of (image array, timestamp) or None
        """
        # Try to get from cache first (non-blocking, fastest)
        try:
            # Get all available frames and keep only the latest
            latest = None
            while True:
                try:
                    latest = self._image_cache.get_nowait()
                except queue.Empty:
                    break
            if latest:
                # Record image if recording is enabled
                if self._recorder and self._recorder.is_recording():
                    img, ts = latest
                    if HAS_NUMPY and img is not None:
                        self._recorder.record_image(img, ts)
                return latest
        except Exception:
            pass
        
        # Fallback to legacy latest
        with self._lock:
            latest = getattr(self, "_latest_image", None)
            if latest and self._recorder and self._recorder.is_recording():
                img, ts = latest
                if HAS_NUMPY and img is not None:
                    self._recorder.record_image(img, ts)
            return latest
            
    def get_latest_point_cloud(self) -> Optional[Tuple]:
        """
        Get the latest mock point cloud from cache (non-blocking).
        
        Returns:
            Tuple of (points array, timestamp) or None
        """
        # Try to get from cache first (non-blocking, fastest)
        try:
            # Get all available frames and keep only the latest
            latest = None
            while True:
                try:
                    latest = self._pointcloud_cache.get_nowait()
                except queue.Empty:
                    break
            if latest:
                # Record point cloud if recording is enabled
                if self._recorder and self._recorder.is_recording():
                    points, ts = latest
                    if HAS_NUMPY and points is not None:
                        self._recorder.record_pointcloud(points, ts)
                return latest
        except Exception:
            pass
        
        # Fallback to legacy latest
        with self._lock:
            latest = getattr(self, "_latest_point_cloud", None)
            if latest and self._recorder and self._recorder.is_recording():
                points, ts = latest
                if HAS_NUMPY and points is not None:
                    self._recorder.record_pointcloud(points, ts)
            return latest
            
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

