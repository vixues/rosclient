"""Production ROS client implementation."""
from __future__ import annotations

import base64
import logging
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as FutureTimeoutError
from typing import Optional, Dict, Any, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
import roslibpy

from ..core.base import RosClientBase
from ..core.topic_service_manager import TopicServiceManager
from ..models.state import ConnectionState
from ..utils.backoff import exponential_backoff
from ..utils.logger import setup_logger
from .config import DEFAULT_CONFIG, DEFAULT_TOPICS


class RosClient(RosClientBase):
    """Production ROS client using roslibpy."""

    def __init__(self, connection_str: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ROS client.
        
        Args:
            connection_str: WebSocket URL (e.g., "ws://host:port")
            config: Optional configuration dictionary
        """
        super().__init__(connection_str, config=config)

        # Apply default configuration
        for k, v in DEFAULT_CONFIG.items():
            self._config.setdefault(k, v)

        # parse/validate websocket URL more robustly
        parsed = urlparse(self.connection_str)
        if parsed.scheme not in ("ws", "wss") or not parsed.hostname:
            raise ValueError(f"Invalid WebSocket URL: {self.connection_str}")
        self._host = parsed.hostname
        self._port = parsed.port or 9090
        self._path = parsed.path or "/"

        self._ros: Optional[roslibpy.Ros] = None
        self._ts_mgr: Optional[TopicServiceManager] = None
        
        # High-frequency cache for images and point clouds
        # Use queues with maxsize to keep only the latest frames (drop old ones)
        self._image_cache: queue.Queue = queue.Queue(maxsize=3)  # Keep latest 3 frames
        self._pointcloud_cache: queue.Queue = queue.Queue(maxsize=3)  # Keep latest 3 frames
        
        # Legacy support - keep latest for backward compatibility
        self._latest_image: Optional[Tuple[np.ndarray, float]] = None
        self._latest_point_cloud: Optional[Tuple[np.ndarray, float]] = None

        self.log = setup_logger(f"RosClient[{self._host}:{self._port}]")
        self.log.setLevel(self._config.get("logger_level", logging.INFO))

    def _ensure_ts_mgr(self) -> TopicServiceManager:
        """
        Ensure TopicServiceManager is initialized.
        
        Returns:
            TopicServiceManager instance
            
        Raises:
            RuntimeError: If not connected
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to ROS")
        if self._ts_mgr is None:
            raise RuntimeError("TopicServiceManager not initialized")
        return self._ts_mgr

    def connect_async(self) -> None:
        """Asynchronously connect to ROS with retry logic."""
        def task():
            with self._connect_lock:
                if self._connecting:
                    self.log.debug("Connect already in progress, skipping")
                    return
                self._connecting = True

            try:
                max_retries = int(self._config.get("connect_max_retries", 5))
                base_backoff = float(self._config.get("connect_backoff_seconds", 1.0))
                max_backoff = float(self._config.get("connect_backoff_max", 30.0))

                for attempt in range(1, max_retries + 1):
                    if self._stop.is_set():
                        self.log.info("Stop requested, aborting connect attempts")
                        return
                    if self._connection_state == ConnectionState.CONNECTED:
                        self.log.debug("Already connected, skipping connect attempt")
                        return
                    try:
                        self.log.info(f"Connecting to ws://{self._host}:{self._port} (attempt {attempt}/{max_retries})")
                        ros = roslibpy.Ros(host=self._host, port=self._port)
                        ros.run()
                        if getattr(ros, "is_connected", False):
                            with self._lock:
                                self._ros = ros
                                self._ts_mgr = TopicServiceManager(ros, f"{self._host}:{self._port}")
                                self._state.connected = True
                                self._connection_state = ConnectionState.CONNECTED
                            self.log.info("Connected to rosbridge successfully")
                            try:
                                self._subscribe_topics()
                            except Exception as e:
                                self.log.warning(f"Subscription setup failed: {e}")
                            return
                        else:
                            self.log.warning("roslibpy reported not connected after run()")
                    except Exception as e:
                        self.log.warning(f"Connect attempt {attempt} failed: {e}", exc_info=True)

                    sleep_time = exponential_backoff(base_backoff, attempt, max_backoff)
                    self.log.debug(f"Sleeping {sleep_time:.2f}s before next connect attempt")
                    time.sleep(sleep_time)

                self.log.critical("Failed to connect after maximum retries")
            finally:
                with self._connect_lock:
                    self._connecting = False

        t = threading.Thread(target=task, daemon=True, name=f"Connect-{self._host}:{self._port}")
        t.start()

    def terminate(self) -> None:
        """Terminate the ROS connection."""
        self._stop.set()
        with self._lock:
            if self._ts_mgr:
                try:
                    self._ts_mgr.close_all()
                except Exception as e:
                    self.log.warning(f"Failed to close TopicServiceManager: {e}")
            if self._ros:
                try:
                    terminate_fn = getattr(self._ros, "terminate", None)
                    if callable(terminate_fn):
                        terminate_fn()
                        self.log.info("Terminated ROS connection.")
                except Exception as e:
                    self.log.warning(f"Terminate failed: {e}")
            self._ros = None
            self._ts_mgr = None
            self._state.connected = False
            self._connection_state = ConnectionState.DISCONNECTED
        self.log.debug("RosClient terminated and cleaned up.")

    def _subscribe_topics(self) -> None:
        """Subscribe to default ROS topics."""
        if not self.is_connected():
            self.log.warning("Cannot subscribe — not connected.")
            return
        try:
            self.log.info("Subscribing to state/battery/gps topics.")
            self._ts_mgr.topic(DEFAULT_TOPICS["state"].name, DEFAULT_TOPICS["state"].type).subscribe(self.update_state)
            self._ts_mgr.topic(DEFAULT_TOPICS["battery"].name, DEFAULT_TOPICS["battery"].type).subscribe(self.update_battery)
            self._ts_mgr.topic(DEFAULT_TOPICS["gps"].name, DEFAULT_TOPICS["gps"].type).subscribe(self.update_gps)
            self._ts_mgr.topic(DEFAULT_TOPICS["odom"].name, DEFAULT_TOPICS["odom"].type).subscribe(self.update_odom)
            self._ts_mgr.topic(DEFAULT_TOPICS["drone_state"].name, DEFAULT_TOPICS["drone_state"].type).subscribe(self.update_drone_state)
            cam_name = self._config.get("camera_topic", DEFAULT_TOPICS["camera"].name)
            cam_type = self._config.get("camera_type", DEFAULT_TOPICS["camera"].type)
            self._ts_mgr.topic(cam_name, cam_type).subscribe(self.update_camera)
            self._ts_mgr.topic(DEFAULT_TOPICS["point_cloud"].name, DEFAULT_TOPICS["point_cloud"].type).subscribe(self.update_point_cloud)
        except Exception as e:
            self.log.exception(f"Failed to subscribe topics: {e}")

    # ---------- topic handlers ----------

    def update_state(self, msg: Dict[str, Any]) -> None:
        """Handle state topic updates."""
        try:
            with self._lock:
                self._state.connected = True
                self._state.armed = bool(msg.get("armed", self._state.armed))
                self._state.mode = str(msg.get("mode", self._state.mode))
                self._state.last_updated = time.time()
            self.log.debug(f"State updated: mode={self._state.mode}, armed={self._state.armed}")
        except Exception as e:
            self.log.exception(f"Error handling state update: {e}")

    def update_drone_state(self, msg: Dict[str, Any]) -> None:
        """Handle drone state topic updates."""
        try:
            with self._lock:
                self._state.landed = bool(msg.get("landed", self._state.landed))
                self._state.returned = bool(msg.get("returned", self._state.returned))
                self._state.reached = bool(msg.get("reached", self._state.reached))
                self._state.tookoff = bool(msg.get("tookoff", self._state.tookoff))
                self._state.last_updated = time.time()
            self.log.debug(f"Drone state updated: landed={self._state.landed}, returned={self._state.returned}")
        except Exception as e:
            self.log.exception(f"Error handling drone state update: {e}")

    def update_camera(self, msg: Dict[str, Any]) -> None:
        """Receive camera image messages and convert to OpenCV format."""
        try:
            # Log comprehensive message metadata for diagnostics
            msg_keys = list(msg.keys())
            data_type = type(msg.get("data")).__name__ if "data" in msg else "N/A"
            data_len = len(msg.get("data")) if isinstance(msg.get("data"), (bytes, bytearray, str)) else "N/A"
            self.log.info(f"Received camera message: keys={msg_keys}, data_type={data_type}, data_len={data_len}")
        except Exception as e:
            self.log.warning(f"Failed to log camera message metadata: {e}")
        
        try:
            frame = None
            # CompressedImage: typically has 'data' with base64 or raw bytes
            if "data" in msg and isinstance(msg["data"], (bytes, bytearray)):
                self.log.info(f"Camera payload branch: raw bytes, size={len(msg['data'])} bytes")
                np_arr = np.frombuffer(msg["data"], np.uint8)
                self.log.info(f"Numpy array shape: {np_arr.shape}")
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    self.log.info(f"Successfully decoded frame: shape={frame.shape}, dtype={frame.dtype}")
                else:
                    self.log.warning("cv2.imdecode returned None for raw bytes payload")
            elif "data" in msg and isinstance(msg["data"], str):
                self.log.info(f"Camera payload branch: base64 string, encoded_size={len(msg['data'])} chars")
                try:
                    # Step 1: Base64 decode
                    self.log.info(f"Step 1: Starting base64 decode from {len(msg['data'])} characters")
                    img_data = base64.b64decode(msg["data"]) if msg["data"] else b""
                    self.log.info(f"Step 1 result: Base64 decoded successfully to {len(img_data)} bytes")

                    # Step 2: Analyze payload signature
                    self.log.info(f"Step 2: Analyzing payload signature (first 20 bytes)")
                    hex_sig = img_data[:20].hex()
                    self.log.info(f"Step 2 result: Payload signature (hex): {hex_sig}")

                    # Detect format from magic bytes
                    if img_data.startswith(b'\xff\xd8\xff'):
                        format_detected = "JPEG"
                    elif img_data.startswith(b'\x89PNG'):
                        format_detected = "PNG"
                    elif img_data.startswith(b'BM'):
                        format_detected = "BMP"
                    elif img_data.startswith(b'GIF'):
                        format_detected = "GIF"
                    elif img_data.startswith(b'RIFF') and img_data[8:12] == b'WEBP':
                        format_detected = "WebP"
                    else:
                        format_detected = "RAW"
                    self.log.info(f"Step 2 detail: Detected format from magic bytes: {format_detected}")

                    # Step 3: Convert to numpy array
                    self.log.info(f"Step 3: Converting decoded bytes to numpy array (uint8)")
                    np_arr = np.frombuffer(img_data, dtype=np.uint8)
                    self.log.info(f"Step 3 result: Numpy array created - shape={np_arr.shape}, dtype={np_arr.dtype}, size={np_arr.size} bytes")

                    # Step 4: Attempt cv2.imdecode if it's compressed
                    frame = None
                    if format_detected in ["JPEG", "PNG", "BMP", "GIF", "WebP"]:
                        self.log.info(f"Step 4: Attempting cv2.imdecode with IMREAD_COLOR flag")
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    # Step 5: If imdecode failed or RAW format, try manual reshape
                    if frame is None:
                        self.log.warning(f"Step 4 result: cv2.imdecode returned None, attempting manual reshape")
                        try:
                            # 需要知道 width, height, channels
                            width = msg.get("width")
                            height = msg.get("height")
                            encoding = msg.get("encoding", "bgr8")
                            channels = 3 if encoding in ["bgr8", "rgb8"] else 1

                            if width and height:
                                frame = np_arr.reshape((height, width, channels))
                                if encoding == "rgb8":
                                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                self.log.info(f"Step 5 result: Successfully reshaped raw data to frame")
                                self.log.info(f"Frame shape={frame.shape}, dtype={frame.dtype}, size={frame.nbytes} bytes")
                            else:
                                self.log.error(f"Cannot reshape raw data: missing width/height info")
                        except Exception as reshape_err:
                            self.log.error(f"Manual reshape failed: {type(reshape_err).__name__}: {reshape_err}", exc_info=True)
                    else:
                        self.log.info(f"Step 4 result: cv2.imdecode successful, frame ready")
                
                except Exception as b64_err:
                    self.log.error(f"Base64 decoding exception: {type(b64_err).__name__}: {b64_err}", exc_info=True)

            elif "encoding" in msg and "data" in msg:
                height = int(msg.get("height", 0))
                width = int(msg.get("width", 0))
                encoding = msg.get("encoding", "bgr8")
                channels = 3 if encoding in ("rgb8", "bgr8") else 1
                self.log.info(f"Camera payload branch: raw buffer, encoding={encoding}, dims={width}x{height}, channels={channels}")
                img_data = np.frombuffer(msg["data"], dtype=np.uint8)
                expected_size = height * width * channels
                actual_size = img_data.size
                self.log.info(f"Buffer size check: expected={expected_size}, actual={actual_size}")
                if height > 0 and width > 0 and actual_size == expected_size:
                    frame = img_data.reshape((height, width, channels))
                    self.log.info(f"Successfully reshaped frame: shape={frame.shape}, dtype={frame.dtype}")
                else:
                    self.log.warning(f"Raw image shape mismatch: expected {expected_size} bytes, got {actual_size} bytes")
            else:
                self.log.warning(f"Received unknown camera message format. Message keys: {list(msg.keys())}")
                return

            if frame is None:
                self.log.warning("Failed to decode camera frame (frame is None after all branches)")
                return

            timestamp = time.time()
            
            # Update cache (non-blocking, drop old frames if queue is full)
            try:
                self._image_cache.put_nowait((frame, timestamp))
                self.log.info(f"Added frame to cache at {timestamp:.3f}s")
            except queue.Full:
                # Queue is full, remove oldest and add new
                try:
                    old_frame = self._image_cache.get_nowait()
                    self._image_cache.put_nowait((frame, timestamp))
                    self.log.debug(f"Cache full, dropped old frame and added new frame at {timestamp:.3f}s")
                except queue.Empty:
                    self.log.warning("Cache full but was empty when trying to remove frame")
            
            # Update legacy latest for backward compatibility
            with self._lock:
                self._latest_image = (frame, timestamp)
                self._state.last_updated = timestamp

            self.log.info(f"Camera frame successfully processed and cached: shape={frame.shape} at {timestamp:.3f}s")
        except Exception as e:
            self.log.exception(f"Error decoding camera image: {e}", exc_info=True)

    def fetch_camera_image(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        Pull one image from camera topic synchronously.
        
        Returns:
            Tuple of (image, timestamp) or None if unavailable
        """
        try:
            with self._lock:
                if not self.is_connected():
                    self.log.warning("Cannot fetch image — not connected to ROS.")
                    return None
                cam_name = self._config.get("camera_topic", DEFAULT_TOPICS["camera"].name)
                cam_type = self._config.get("camera_type", DEFAULT_TOPICS["camera"].type)
                topic = self._ts_mgr.topic(cam_name, cam_type)

            msg = topic.get_message(timeout=3.0)
            if not msg:
                self.log.warning("No image message received from ROS.")
                return None

            # decode as in update_camera (more verbose logging for diagnostics)
            frame = None
            if "data" in msg and isinstance(msg["data"], (bytes, bytearray)):
                self.log.debug("fetch_camera_image: raw bytes payload")
                np_arr = np.frombuffer(msg["data"], np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            elif "data" in msg and isinstance(msg["data"], str):
                self.log.debug("fetch_camera_image: base64 payload (decoding)")
                try:
                    # Step 1: Base64 decode
                    self.log.debug(f"fetch_camera_image Step 1: Base64 decode from {len(msg['data'])} chars")
                    img_data = base64.b64decode(msg["data"]) if msg["data"] else b""
                    self.log.debug(f"fetch_camera_image Step 1: Decoded to {len(img_data)} bytes")
                    
                    # Step 2: Signature analysis
                    self.log.debug(f"fetch_camera_image Step 2: Payload signature (hex): {img_data[:20].hex()}")
                    if img_data.startswith(b'\xff\xd8\xff'):
                        fmt = "JPEG"
                    elif img_data.startswith(b'\x89PNG'):
                        fmt = "PNG"
                    else:
                        fmt = "OTHER"
                    self.log.debug(f"fetch_camera_image Step 2: Detected format: {fmt}")
                    
                    # Step 3: Convert to numpy array
                    np_arr = np.frombuffer(img_data, np.uint8)
                    self.log.debug(f"fetch_camera_image Step 3: Numpy array shape={np_arr.shape}, min={np_arr.min()}, max={np_arr.max()}")
                    
                    # Step 4: Decode with cv2
                    self.log.debug(f"fetch_camera_image Step 4: Attempting cv2.imdecode(IMREAD_COLOR)")
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        self.log.warning(f"fetch_camera_image Step 4: IMREAD_COLOR failed, trying IMREAD_UNCHANGED")
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                    
                    if frame is not None:
                        self.log.debug(f"fetch_camera_image Step 4: SUCCESS - frame shape={frame.shape}, dtype={frame.dtype}")
                    else:
                        self.log.error(f"fetch_camera_image Step 4: FAILED - both decode flags returned None (format: {fmt})")
                except Exception as e:
                    self.log.error(f"fetch_camera_image base64 error: {type(e).__name__}: {e}", exc_info=True)
                    return None
            elif "encoding" in msg and "data" in msg:
                h = int(msg.get("height", 0))
                w = int(msg.get("width", 0))
                encoding = msg.get("encoding", "bgr8")
                channels = 3 if encoding in ("rgb8", "bgr8") else 1
                img_data = np.frombuffer(msg["data"], dtype=np.uint8)
                if h > 0 and w > 0 and img_data.size == h * w * channels:
                    frame = img_data.reshape((h, w, channels))
                else:
                    self.log.warning("Image raw data size doesn't match dimensions")
                    return None
            else:
                self.log.warning("Unknown image message format.")
                return None

            timestamp = time.time()
            self.log.debug(f"Fetched image frame at {timestamp:.3f}s")
            return frame, timestamp

        except Exception as e:
            self.log.exception(f"Error fetching camera image: {e}")
            return None

    def get_latest_image(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        Get the latest received image from cache (non-blocking).
        
        Returns:
            Tuple of (image, timestamp) or None
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
                return latest
        except Exception:
            pass
        
        # Fallback to legacy latest
        with self._lock:
            return getattr(self, "_latest_image", None)

    def get_latest_point_cloud(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        Get the latest received point cloud from cache (non-blocking).
        
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
                return latest
        except Exception:
            pass
        
        # Fallback to legacy latest
        with self._lock:
            return getattr(self, "_latest_point_cloud", None)

    def update_point_cloud(self, msg: Dict[str, Any]) -> None:
        """Handle point cloud topic updates."""
        try:
            result = self._decode_point_cloud(msg)
            if result:
                points, ts = result
                
                # Update cache (non-blocking, drop old frames if queue is full)
                try:
                    self._pointcloud_cache.put_nowait((points, ts))
                except queue.Full:
                    # Queue is full, remove oldest and add new
                    try:
                        self._pointcloud_cache.get_nowait()
                        self._pointcloud_cache.put_nowait((points, ts))
                    except queue.Empty:
                        pass
                
                # Update legacy latest for backward compatibility
                with self._lock:
                    self._latest_point_cloud = (points, ts)
                    self._state.last_updated = ts
                self.log.debug(f"Received point cloud frame with {len(points)} points at {ts:.3f}s")
        except Exception as e:
            self.log.exception(f"Error handling point cloud update: {e}")

    def fetch_point_cloud(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        Fetch point cloud data synchronously.
        
        Returns:
            Tuple of (points array, timestamp) or None
        """
        try:
            with self._lock:
                if not self.is_connected():
                    self.log.warning("Cannot fetch point cloud — not connected to ROS.")
                    return None

                conf = self._config
                topic_name = conf.get("point_cloud_topic", "/drone_1_cloud_registered")
                topic_type = conf.get("point_cloud_type", "sensor_msgs/PointCloud2")

                topic = self._ensure_ts_mgr().topic(topic_name, topic_type)

            msg = topic.get_message(timeout=3.0)
            if not msg:
                self.log.warning("No point cloud message received from ROS.")
                return None

            if "data" not in msg or "fields" not in msg:
                self.log.warning("Invalid PointCloud2 message format.")
                return None

            result = self._decode_point_cloud(msg)
            if not result:
                self.log.warning("Failed to decode point cloud.")
                return None

            points, ts = result
            self.log.debug(f"Fetched point cloud with {len(points)} points at {ts:.3f}s")
            return points, ts

        except Exception as e:
            self.log.exception(f"Error fetching point cloud: {e}")
            return None

    def update_battery(self, msg: Dict[str, Any]) -> None:
        """Handle battery topic updates."""
        try:
            with self._lock:
                p = msg.get("percentage", msg.get("percent", msg.get("battery", 1.0)))
                try:
                    p_val = float(p)
                    self._state.battery = (p_val * 100.0) if p_val <= 1.0 else p_val
                except Exception:
                    self.log.debug("Unable to parse battery percentage; leaving previous value")
                self._state.last_updated = time.time()
            self.log.debug(f"Battery: {self._state.battery:.1f}%")
        except Exception as e:
            self.log.exception(f"Error handling battery update: {e}")

    def update_gps(self, msg: Dict[str, Any]) -> None:
        """Handle GPS topic updates."""
        try:
            with self._lock:
                try:
                    self._state.latitude = float(msg.get("latitude", msg.get("lat", self._state.latitude)))
                    self._state.longitude = float(msg.get("longitude", msg.get("lon", self._state.longitude)))
                    self._state.altitude = float(msg.get("altitude", msg.get("alt", self._state.altitude)))
                except Exception:
                    self.log.debug("Partial or invalid GPS data received")
                self._state.last_updated = time.time()
            self.log.debug(f"GPS: lat={self._state.latitude:.6f}, lon={self._state.longitude:.6f}")
        except Exception as e:
            self.log.exception(f"Error handling GPS update: {e}")

    def service_call(self, service_name: str, service_type: str, payload: Dict[str, Any],
                          timeout: Optional[float] = None, retries: Optional[int] = None) -> Dict[str, Any]:
        """
        Call a ROS service with retry logic.
        
        Args:
            service_name: Service name
            service_type: Service type
            payload: Service request payload
            timeout: Timeout in seconds
            retries: Number of retries
            
        Returns:
            Service response dictionary
            
        Raises:
            Exception: If all retries fail
        """
        timeout = timeout or float(self._config.get("service_call_timeout", 5.0))
        retries = retries if retries is not None else int(self._config.get("service_call_retries", 2))
        last_exc: Optional[Exception] = None

        for attempt in range(1, retries + 2):
            try:
                svc = self._ensure_ts_mgr().service(service_name, service_type)
                req = roslibpy.ServiceRequest(payload or {})
                self.log.debug(f"Calling service {service_name} attempt {attempt} payload={payload}")
                with ThreadPoolExecutor(max_workers=1) as ex:
                    fut: Future = ex.submit(svc.call, req)
                    resp = fut.result(timeout=timeout)
                self.log.debug(f"Service {service_name} response: {resp}")
                return resp or {}
            except FutureTimeoutError:
                last_exc = TimeoutError(f"Service call {service_name} timed out after {timeout}s")
                self.log.warning(str(last_exc))
            except Exception as e:
                last_exc = e
                self.log.warning(f"Service call {service_name} failed on attempt {attempt}: {e}")
            if attempt <= retries:
                backoff = min(5.0, 0.5 * (2 ** (attempt - 1)))
                jitter = random.uniform(0, backoff * 0.2)
                time.sleep(backoff + jitter)

        raise last_exc if last_exc is not None else RuntimeError("Unknown service call failure")

    def publish(self, topic_name: str, topic_type: str, message: Dict[str, Any],
                     retries: Optional[int] = None) -> None:
        """
        Publish to a ROS topic with retry logic.
        
        Args:
            topic_name: Topic name
            topic_type: Topic type
            message: Message dictionary
            retries: Number of retries
            
        Raises:
            Exception: If all retries fail
        """
        retries = retries if retries is not None else int(self._config.get("publish_retries", 2))
        last_exc: Optional[Exception] = None

        for attempt in range(1, retries + 2):
            try:
                topic = self._ensure_ts_mgr().topic(topic_name, topic_type)
                topic.publish(roslibpy.Message(message))
                self.log.debug(f"Published to {topic_name} (attempt {attempt}): {message}")
                return
            except Exception as e:
                last_exc = e
                self.log.warning(f"Publish to {topic_name} failed on attempt {attempt}: {e}")
            if attempt <= retries:
                time.sleep(0.2 * attempt)

        self.log.error(f"Failed to publish to {topic_name} after {retries + 1} attempts: {last_exc}")
        raise last_exc if last_exc is not None else RuntimeError("Unknown publish failure")

