"""ROS bag file playback client."""
from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Callable

try:
    import rosbag
    HAS_ROSBAG = True
except ImportError:
    HAS_ROSBAG = False
    rosbag = None

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
    HAS_NUMPY = True
except ImportError:
    HAS_CV2 = False
    HAS_NUMPY = False
    np = None

from ..core.base import RosClientBase
from ..models.state import ConnectionState
from ..utils.logger import setup_logger


class RosbagClient(RosClientBase):
    """Client for playing back ROS bag files."""
    
    def __init__(self, bag_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ROS bag client.
        
        Args:
            bag_path: Path to the ROS bag file
            config: Optional configuration dictionary
        """
        super().__init__(f"rosbag://{bag_path}", config=config)
        
        if not HAS_ROSBAG:
            raise ImportError("rosbag module not available. Install with: pip install rospy")
        
        self.bag_path = Path(bag_path)
        if not self.bag_path.exists():
            raise FileNotFoundError(f"Bag file not found: {bag_path}")
        
        self.bag: Optional[rosbag.Bag] = None
        self._playback_thread: Optional[threading.Thread] = None
        self._stop_playback = threading.Event()
        self._playback_paused = threading.Event()
        self._playback_paused.set()  # Start paused
        
        # Playback state
        self._playback_speed = 1.0  # 1.0 = real-time, 2.0 = 2x speed, etc.
        self._current_time = 0.0
        self._start_time = None
        self._bag_start_time = None
        self._bag_duration = 0.0
        
        # Topic subscriptions (callbacks)
        self._topic_callbacks: Dict[str, List[Callable]] = {}
        
        # Message cache (latest message per topic)
        self._message_cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_lock = threading.Lock()
        
        # Image and point cloud caches
        self._image_cache: queue.Queue = queue.Queue(maxsize=3)
        self._pointcloud_cache: queue.Queue = queue.Queue(maxsize=3)
        self._latest_image: Optional[Tuple[np.ndarray, float]] = None
        self._latest_point_cloud: Optional[Tuple[np.ndarray, float]] = None
        
        self.log = setup_logger(f"RosbagClient[{self.bag_path.name}]")
        self.log.setLevel(self._config.get("logger_level", logging.INFO))
        
        # Load bag info
        self._load_bag_info()
    
    def _load_bag_info(self):
        """Load bag file information."""
        try:
            self.bag = rosbag.Bag(str(self.bag_path), 'r')
            
            # Get bag start time and duration
            bag_info = self.bag.get_type_and_topic_info()
            if bag_info.message_count > 0:
                # Get first and last message times
                start_time = None
                end_time = None
                for topic, msg, t in self.bag.read_messages():
                    if start_time is None:
                        start_time = t
                    end_time = t
                if start_time and end_time:
                    self._bag_start_time = start_time.to_sec()
                    self._bag_duration = (end_time.to_sec() - self._bag_start_time)
            
            self.log.info(f"Loaded bag file: {self.bag_path.name}, duration: {self._bag_duration:.2f}s")
        except Exception as e:
            self.log.error(f"Failed to load bag file: {e}")
            raise
    
    def connect_async(self) -> None:
        """Connect (open bag file)."""
        if self._connection_state == ConnectionState.CONNECTED:
            return
        
        try:
            if self.bag is None:
                self._load_bag_info()
            
            with self._lock:
                self._connection_state = ConnectionState.CONNECTED
                self._state.connected = True
            
            self.log.info("Bag file opened successfully")
        except Exception as e:
            self.log.error(f"Failed to open bag file: {e}")
            with self._lock:
                self._connection_state = ConnectionState.DISCONNECTED
                self._state.connected = False
    
    def is_connected(self) -> bool:
        """Check if bag file is open."""
        return (self._connection_state == ConnectionState.CONNECTED and 
                self.bag is not None)
    
    def disconnect(self) -> None:
        """Disconnect (close bag file)."""
        self.stop_playback()
        if self.bag:
            try:
                self.bag.close()
            except:
                pass
            self.bag = None
        
        with self._lock:
            self._connection_state = ConnectionState.DISCONNECTED
            self._state.connected = False
    
    def terminate(self) -> None:
        """Terminate and cleanup."""
        self.disconnect()
        super().terminate()
    
    def start_playback(self, speed: float = 1.0):
        """Start playing back the bag file."""
        if not self.is_connected():
            self.log.warning("Cannot start playback - not connected")
            return
        
        if self._playback_thread and self._playback_thread.is_alive():
            self.log.warning("Playback already running")
            return
        
        self._playback_speed = speed
        self._stop_playback.clear()
        self._playback_paused.clear()
        self._start_time = time.time()
        
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._playback_thread.start()
        self.log.info(f"Started playback at {speed}x speed")
    
    def stop_playback(self):
        """Stop playback."""
        self._stop_playback.set()
        self._playback_paused.set()
        if self._playback_thread:
            self._playback_thread.join(timeout=2.0)
        self._current_time = 0.0
        self.log.info("Stopped playback")
    
    def pause_playback(self):
        """Pause playback."""
        self._playback_paused.set()
        self.log.info("Paused playback")
    
    def resume_playback(self):
        """Resume playback."""
        self._playback_paused.clear()
        self.log.info("Resumed playback")
    
    def set_playback_speed(self, speed: float):
        """Set playback speed (1.0 = real-time, 2.0 = 2x, etc.)."""
        self._playback_speed = max(0.1, min(10.0, speed))
        self.log.info(f"Playback speed set to {self._playback_speed}x")
    
    def seek_to_time(self, time_offset: float):
        """Seek to a specific time offset in the bag (in seconds from start)."""
        self._current_time = max(0.0, min(time_offset, self._bag_duration))
        if self._start_time:
            self._start_time = time.time() - (self._current_time / self._playback_speed)
        self.log.info(f"Seeked to {self._current_time:.2f}s")
    
    def get_current_time(self) -> float:
        """Get current playback time in seconds."""
        return self._current_time
    
    def get_duration(self) -> float:
        """Get total bag duration in seconds."""
        return self._bag_duration
    
    def get_progress(self) -> float:
        """Get playback progress (0.0 to 1.0)."""
        if self._bag_duration == 0:
            return 0.0
        return self._current_time / self._bag_duration
    
    def subscribe(self, topic: str, callback: Callable):
        """Subscribe to a topic in the bag file."""
        if topic not in self._topic_callbacks:
            self._topic_callbacks[topic] = []
        self._topic_callbacks[topic].append(callback)
        self.log.info(f"Subscribed to topic: {topic}")
    
    def _playback_loop(self):
        """Main playback loop."""
        if not self.bag:
            return
        
        try:
            # Reset bag to beginning
            self.bag.close()
            self.bag = rosbag.Bag(str(self.bag_path), 'r')
            
            # Calculate start offset
            start_offset = self._current_time
            target_time = self._bag_start_time + start_offset
            
            # Read messages
            for topic, msg, t in self.bag.read_messages(start_time=rosbag.rostime.Time.from_sec(target_time)):
                if self._stop_playback.is_set():
                    break
                
                # Wait if paused
                while self._playback_paused.is_set() and not self._stop_playback.is_set():
                    time.sleep(0.1)
                
                if self._stop_playback.is_set():
                    break
                
                # Calculate message time relative to bag start
                msg_time = t.to_sec() - self._bag_start_time
                
                # Update current time
                self._current_time = msg_time
                
                # Calculate when to process this message (real-time playback)
                if self._start_time:
                    real_time = time.time()
                    expected_time = self._start_time + (msg_time / self._playback_speed)
                    sleep_time = (expected_time - real_time) / self._playback_speed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                # Process message
                self._process_message(topic, msg, msg_time)
                
                # Check if we've reached the end
                if self._current_time >= self._bag_duration:
                    self.log.info("Reached end of bag file")
                    break
                    
        except Exception as e:
            self.log.error(f"Playback error: {e}")
        finally:
            if self.bag:
                try:
                    self.bag.close()
                except:
                    pass
    
    def _process_message(self, topic: str, msg: Any, timestamp: float):
        """Process a message from the bag file."""
        # Update cache
        with self._cache_lock:
            self._message_cache[topic] = (msg, timestamp)
        
        # Call callbacks
        if topic in self._topic_callbacks:
            for callback in self._topic_callbacks[topic]:
                try:
                    callback(msg)
                except Exception as e:
                    self.log.warning(f"Error in callback for {topic}: {e}")
        
        # Handle special topics (images, point clouds)
        self._handle_special_topics(topic, msg, timestamp)
    
    def _handle_special_topics(self, topic: str, msg: Any, timestamp: float):
        """Handle special topic types (images, point clouds)."""
        # Handle image topics
        if hasattr(msg, 'data') and hasattr(msg, 'encoding') and hasattr(msg, 'width') and hasattr(msg, 'height'):
            try:
                if HAS_CV2:
                    # Decode image
                    img_data = bytes(msg.data)
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        self._latest_image = (img, timestamp)
                        try:
                            self._image_cache.put_nowait((img, timestamp))
                        except queue.Full:
                            try:
                                self._image_cache.get_nowait()
                                self._image_cache.put_nowait((img, timestamp))
                            except queue.Empty:
                                pass
            except Exception as e:
                self.log.debug(f"Error processing image from {topic}: {e}")
        
        # Handle point cloud topics
        if hasattr(msg, 'data') and hasattr(msg, 'points') and hasattr(msg, 'fields'):
            try:
                if HAS_NUMPY:
                    # Extract point cloud data
                    # This is a simplified version - actual implementation depends on PointCloud2 format
                    points = self._extract_point_cloud(msg)
                    if points is not None and len(points) > 0:
                        self._latest_point_cloud = (points, timestamp)
                        try:
                            self._pointcloud_cache.put_nowait((points, timestamp))
                        except queue.Full:
                            try:
                                self._pointcloud_cache.get_nowait()
                                self._pointcloud_cache.put_nowait((points, timestamp))
                            except queue.Empty:
                                pass
            except Exception as e:
                self.log.debug(f"Error processing point cloud from {topic}: {e}")
    
    def _extract_point_cloud(self, msg) -> Optional[np.ndarray]:
        """Extract point cloud data from PointCloud2 message."""
        # Simplified point cloud extraction
        # Full implementation would properly parse PointCloud2 format
        try:
            if hasattr(msg, 'data') and len(msg.data) > 0:
                # This is a placeholder - actual implementation needs proper PointCloud2 parsing
                return None
        except:
            pass
        return None
    
    # Implement RosClientBase interface methods
    def get_status(self):
        """Get current status (compatible with RosClient interface)."""
        return self._state
    
    def get_position(self) -> Tuple[float, float, float]:
        """Get current position from cached messages."""
        # Try to get position from odometry or GPS messages
        with self._cache_lock:
            for topic, (msg, _) in self._message_cache.items():
                if 'odom' in topic.lower() or 'gps' in topic.lower():
                    try:
                        if hasattr(msg, 'pose') and hasattr(msg.pose, 'position'):
                            pos = msg.pose.position
                            return (pos.x, pos.y, pos.z)
                        elif hasattr(msg, 'latitude') and hasattr(msg, 'longitude'):
                            return (msg.latitude, msg.longitude, getattr(msg, 'altitude', 0.0))
                    except:
                        pass
        return (0.0, 0.0, 0.0)
    
    def get_orientation(self) -> Tuple[float, float, float]:
        """Get current orientation from cached messages."""
        import math
        with self._cache_lock:
            for topic, (msg, _) in self._message_cache.items():
                if 'odom' in topic.lower():
                    try:
                        if hasattr(msg, 'pose') and hasattr(msg.pose, 'orientation'):
                            q = msg.pose.orientation
                            # Convert quaternion to Euler angles (simplified)
                            # Roll (x-axis rotation)
                            sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
                            cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
                            roll = math.atan2(sinr_cosp, cosr_cosp)
                            
                            # Pitch (y-axis rotation)
                            sinp = 2 * (q.w * q.y - q.z * q.x)
                            if abs(sinp) >= 1:
                                pitch = math.copysign(math.pi / 2, sinp)
                            else:
                                pitch = math.asin(sinp)
                            
                            # Yaw (z-axis rotation)
                            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
                            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
                            yaw = math.atan2(siny_cosp, cosy_cosp)
                            
                            return (float(roll), float(pitch), float(yaw))
                    except:
                        pass
        return (0.0, 0.0, 0.0)
    
    def get_latest_image(self) -> Optional[Tuple[np.ndarray, float]]:
        """Get latest image from bag file."""
        return self._latest_image
    
    def get_latest_point_cloud(self) -> Optional[Tuple[np.ndarray, float]]:
        """Get latest point cloud from bag file."""
        return self._latest_point_cloud
    
    def publish(self, topic: str, message: Dict[str, Any], message_type: str = "") -> None:
        """Publish message (not supported for bag playback)."""
        self.log.warning("Publish not supported for bag file playback")
    
    def get_topics(self) -> List[str]:
        """Get list of topics in the bag file."""
        if not self.bag:
            return []
        try:
            bag_info = self.bag.get_type_and_topic_info()
            return list(bag_info.topics.keys())
        except:
            return []

