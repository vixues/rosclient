"""AirSim client implementation that maps ROS-style interfaces to AirSim API."""
from __future__ import annotations

import logging
import math
import queue
import threading
import time
from typing import Optional, Dict, Any, Tuple, List

try:
    import airsim
    HAS_AIRSIM = True
except ImportError:
    HAS_AIRSIM = False
    airsim = None

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
from ..processors.image_processor import ImageProcessor
from ..processors.pointcloud_processor import PointCloudProcessor
from ..utils.logger import setup_logger


class AirSimClient(RosClientBase):
    """
    AirSim wrapper that maps common ROS-style services/topics to AirSim API.
    
    Architecture Design:
    - Acts as a mediator/proxy client between GUI and AirSim simulator
    - Separates data acquisition (read) from command execution (write)
    - Data acquisition: High-frequency, non-blocking, real-time (state, images, pointclouds)
    - Command execution: Can block, but doesn't prevent data acquisition
    - Lock strategy: _data_lock for reads (with timeout), _command_lock for writes
    - Ensures data acquisition latency is minimized while commands execute
    
    connection_str: "ip:port" or "ip" (port defaults to 41451)
    """
    
    def __init__(self, connection_str: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AirSim client.
        
        Args:
            connection_str: Connection string (e.g., "ip:port" or "ip")
            config: Optional configuration dictionary
        """
        super().__init__(connection_str, config=config)
        
        if not HAS_AIRSIM:
            raise ImportError("airsim package is not installed. Please install it with 'pip install airsim'")
        
        # Apply default configuration
        self._config.setdefault("control_topic", "/control")
        self._config.setdefault("control_type", "controller_msgs/cmd")
        self._config.setdefault("goal_topic", "/goal_user2brig")
        self._config.setdefault("goal_type", "quadrotor_msgs/GoalSet")
        self._config.setdefault("camera_id", "0")
        self._config.setdefault("image_update_interval", 0.016)  # ~60 FPS for better responsiveness
        self._config.setdefault("pointcloud_update_interval", 0.1)
        self._config.setdefault("service_call_timeout", 10.0)
        self._config.setdefault("service_call_retries", 2)
        self._config.setdefault("publish_retries", 2)
        self._config.setdefault("connect_max_retries", 3)
        self._config.setdefault("connect_backoff_seconds", 1.0)
        self._config.setdefault("reset_on_terminate", False)  # Don't reset simulation by default
        self._config.setdefault("image_retry_attempts", 2)  # Retry attempts for image fetching
        
        self._client: Optional[airsim.MultirotorClient] = None
        
        # Parse connection string "ip:port"
        try:
            ip, port = connection_str.split(":")
            self._ip = ip
            self._port = int(port)
        except ValueError:
            self._ip = connection_str
            self._port = 41451  # default
        
        if isinstance(self._ip, str) and "://" in self._ip:
            self._ip = self._ip.split("://", 1)[1]
        
        # High-frequency cache for images and point clouds
        # Use queues with maxsize=1 for lowest latency (always get latest frame)
        # This ensures we always have the most recent frame without blocking
        self._image_cache: queue.Queue = queue.Queue(maxsize=1)  # Keep only latest frame (lowest latency)
        self._pointcloud_cache: queue.Queue = queue.Queue(maxsize=1)  # Keep only latest frame
        
        # Legacy support - keep latest for backward compatibility
        self._latest_image: Optional[Tuple[np.ndarray, float]] = None
        self._latest_point_cloud: Optional[Tuple[np.ndarray, float]] = None
        
        # Initialize processors
        self.log = setup_logger(f"AirSimClient[{self._ip}:{self._port}]")
        self.log.setLevel(self._config.get("logger_level", logging.INFO))
        
        if HAS_CV2 and HAS_NUMPY:
            self._image_processor = ImageProcessor(self.log)
        else:
            self._image_processor = None
            self.log.warning("OpenCV or NumPy not available, image processing disabled")
        
        if HAS_NUMPY:
            self._pointcloud_processor = PointCloudProcessor(self.log)
        else:
            self._pointcloud_processor = None
            self.log.warning("NumPy not available, point cloud processing disabled")
        
        # Image update thread
        self._image_update_thread: Optional[threading.Thread] = None
        self._pc_update_thread: Optional[threading.Thread] = None
        self._stop_updates = threading.Event()
        
        # Lock strategy: Separate locks for data access and command execution
        # This ensures data acquisition (state, images) is never blocked by commands
        # 
        # _data_lock: For read operations (state monitoring, image fetching)
        #   - Uses timeout to avoid blocking
        #   - If command is executing, skip this cycle and continue
        # _command_lock: For write operations (takeoff, land, move, etc.)
        #   - Can block, but doesn't prevent data acquisition
        # 
        # Note: AirSim API has thread-safety issues, so we still need locks
        # but we optimize for data acquisition latency
        self._data_lock = threading.RLock()  # For data acquisition (read operations)
        self._command_lock = threading.RLock()  # For command execution (write operations)
        
        # Error tracking for recovery
        self._consecutive_image_errors = 0
        self._max_consecutive_errors = 5
        self._last_image_error_time = 0.0
        self._error_recovery_delay = 1.0  # seconds to wait after errors
        
        # Initialize state
        with self._lock:
            self._state.mode = "MANUAL"
            self._state.connected = False
            self._connection_state = ConnectionState.DISCONNECTED
        
        # Start state monitoring thread
        self._start_state_monitoring()
    
    def _join_async_future(self, future: Any, timeout: Optional[float] = None) -> None:
        """
        Join an AirSim asynchronous result, handling older builds where
        Future.join() does not accept a timeout argument, and IOLoop errors
        in multi-threaded environments.
        """
        if future is None:
            return
        
        if timeout is None:
            timeout = 30.0  # Default timeout to prevent indefinite blocking
        
        try:
            # Try with timeout first
            try:
                future.join(timeout)
            except TypeError:
                # Older AirSim versions don't support timeout
                self.log.debug("Future.join timeout not supported, using blocking join")
                future.join()
        except RuntimeError as e:
            error_msg = str(e)
            # Handle IOLoop errors in multi-threaded environments
            if "IOLoop" in error_msg or "already running" in error_msg.lower():
                self.log.warning(f"IOLoop error in async future join: {e}. This may occur in multi-threaded environments.")
                # Try to wait for completion using polling instead
                try:
                    import time
                    start_time = time.time()
                    while time.time() - start_time < timeout:
                        # Check if future is done (if supported)
                        if hasattr(future, 'is_completed') and future.is_completed():
                            break
                        if hasattr(future, 'done') and future.done():
                            break
                        time.sleep(0.1)
                    # If still not done, log warning but don't raise
                    if hasattr(future, 'is_completed') and not future.is_completed():
                        self.log.warning("Async operation may not have completed due to IOLoop error")
                    elif hasattr(future, 'done') and not future.done():
                        self.log.warning("Async operation may not have completed due to IOLoop error")
                except Exception as poll_error:
                    self.log.warning(f"Error polling future status: {poll_error}")
            else:
                # Re-raise other RuntimeErrors
                raise
        except Exception as e:
            # Log other exceptions but don't crash
            error_msg = str(e)
            if "IOLoop" in error_msg or "already running" in error_msg.lower():
                self.log.warning(f"IOLoop error in async future join: {e}")
            else:
                self.log.error(f"Unexpected error joining async future: {e}")
                raise
    
    def _start_state_monitoring(self, interval: float = 0.2) -> None:
        """Start background thread to monitor AirSim state."""
        def monitor_loop():
            while not self._stop.is_set():
                try:
                    if self._client is None:
                        time.sleep(interval)
                        continue
                    try:
                        # Use data lock with short timeout for state monitoring
                        # If command is executing, skip this cycle (state will be updated next cycle)
                        # This ensures state monitoring never blocks and continues at high frequency
                        if self._data_lock.acquire(timeout=0.05):
                            try:
                                state = self._client.getMultirotorState()
                                gps = state.gps_location
                            finally:
                                self._data_lock.release()
                        else:
                            # Lock is held by command or image operation, skip this cycle
                            # This is expected and normal - we'll get state on next cycle
                            time.sleep(interval)
                            continue
                        update_timestamp = time.time()
                        
                        with self._lock:
                            self._state.connected = True
                            self._connection_state = ConnectionState.CONNECTED
                            
                            # AirSim GPS/position is given in local NED units for pose; use as best-effort
                            try:
                                # AirSim exposes gps_location: latitude, longitude, altitude when GPS enabled
                                self._state.latitude = getattr(gps, "latitude", self._state.latitude)
                                self._state.longitude = getattr(gps, "longitude", self._state.longitude)
                                self._state.altitude = getattr(gps, "altitude", self._state.altitude)
                            except Exception:
                                pass
                            
                            # orientation: AirSim returns Quaternionr object fields .x_val etc or roll_val/pitch_val/yaw_val in some versions
                            orientation = state.kinematics_estimated.orientation
                            # try multiple attribute names defensively
                            roll_val = getattr(orientation, "roll_val", None)
                            pitch_val = getattr(orientation, "pitch_val", None)
                            yaw_val = getattr(orientation, "yaw_val", None)
                            if roll_val is None:
                                # quaternion to euler fallback
                                qx = getattr(orientation, "x_val", 0.0)
                                qy = getattr(orientation, "y_val", 0.0)
                                qz = getattr(orientation, "z_val", 0.0)
                                qw = getattr(orientation, "w_val", 1.0)
                                # quaternion -> euler
                                sinr_cosp = 2.0 * (qw * qx + qy * qz)
                                cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
                                r = math.atan2(sinr_cosp, cosr_cosp)
                                sinp = 2.0 * (qw * qy - qz * qx)
                                if abs(sinp) >= 1:
                                    p = math.copysign(math.pi / 2, sinp)
                                else:
                                    p = math.asin(sinp)
                                siny_cosp = 2.0 * (qw * qz + qx * qy)
                                cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
                                y = math.atan2(siny_cosp, cosy_cosp)
                                roll_val, pitch_val, yaw_val = math.degrees(r), math.degrees(p), math.degrees(y)
                            else:
                                roll_val = math.degrees(roll_val)
                                pitch_val = math.degrees(pitch_val)
                                yaw_val = math.degrees(yaw_val)
                            
                            self._state.roll = roll_val
                            self._state.pitch = pitch_val
                            self._state.yaw = yaw_val
                            
                            # landed state
                            try:
                                landed_state = getattr(state, "landed_state", None)
                                if landed_state is not None and airsim is not None:
                                    self._state.landed = landed_state == airsim.LandedState.Landed
                            except Exception:
                                pass
                            
                            # armed flag (some AirSim builds store .armed)
                            try:
                                self._state.armed = bool(getattr(state, "armed", self._state.armed))
                            except Exception:
                                pass
                            
                            # battery: AirSim doesn't return battery by default -> leave as-is or a simulated value
                            # self._state.battery = <simulate if needed>
                            
                            self._state.last_updated = update_timestamp
                            # Add to state history for synchronization
                            self._add_state_to_history(self._state, update_timestamp)
                            
                            # Record state if recording is enabled
                            if self._recorder and self._recorder.is_recording():
                                self._recorder.record_state(self._state, update_timestamp)
                    except Exception as e:
                        # don't kill monitor on transient errors
                        error_msg = str(e)
                        if "cannot be re-sized" in error_msg or "Existing exports" in error_msg:
                            # Known AirSim issue - wait a bit longer
                            time.sleep(interval * 2)
                            continue
                        self.log.debug(f"AirSim state polling transient error: {e}")
                except Exception as e:
                    self.log.exception(f"State monitoring error: {e}")
                finally:
                    time.sleep(interval)
        
        t = threading.Thread(target=monitor_loop, daemon=True, name=f"AirSim-Monitor-{self._ip}")
        t.start()
    
    def _start_image_updates(self) -> None:
        """Start background thread to fetch images from AirSim."""
        if not HAS_CV2 or not HAS_NUMPY or not self._image_processor:
            return
        
        def image_update_loop():
            camera_id = self._config.get("camera_id", "0")
            interval = self._config.get("image_update_interval", 0.033)
            
            while not self._stop_updates.is_set() and not self._stop.is_set():
                try:
                    if not self._client or not self.is_connected():
                        time.sleep(interval)
                        continue
                    
                    # Check if we need to wait due to recent errors
                    current_time = time.time()
                    if self._consecutive_image_errors > 0:
                        time_since_error = current_time - self._last_image_error_time
                        if time_since_error < self._error_recovery_delay:
                            time.sleep(self._error_recovery_delay - time_since_error)
                            continue
                    
                    # Get image from AirSim (PNG bytes) with retry
                    img_bytes = None
                    max_retries = self._config.get("image_retry_attempts", 2)
                    for retry in range(max_retries):
                        try:
                            img_bytes = self.get_camera_image(camera_id, 0, retries=0)  # 0 = Scene (RGB), retries handled internally
                            if img_bytes:
                                break
                        except Exception as e:
                            error_msg = str(e)
                            if "cannot be re-sized" in error_msg or "Existing exports" in error_msg:
                                # Wait longer for this specific error
                                if retry < max_retries - 1:
                                    time.sleep(0.2 * (retry + 1))
                                    continue
                            if retry < max_retries - 1:
                                time.sleep(0.1)
                            else:
                                raise
                    
                    if img_bytes:
                        # Convert PNG bytes to numpy array
                        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            # Convert BGR to RGB for consistency
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            timestamp = time.time()
                            
                            # Reset error counter on success
                            self._consecutive_image_errors = 0
                            
                            # Synchronize state with image timestamp
                            synced_state = self.sync_state_with_data(timestamp)
                            self._update_state_with_timestamp(timestamp)
                            
                            # Record image with synchronized state if recording is enabled
                            if self._recorder and self._recorder.is_recording():
                                self._recorder.record_image(frame, timestamp, state=synced_state)
                            
                            # Update cache (non-blocking, always replace with latest)
                            # Use put_nowait with get_nowait to ensure we always have the latest frame
                            try:
                                # Remove old frame if exists
                                try:
                                    self._image_cache.get_nowait()
                                except queue.Empty:
                                    pass
                                # Add new frame
                                self._image_cache.put_nowait((frame, timestamp))
                            except queue.Full:
                                # Should not happen with maxsize=1, but handle gracefully
                                try:
                                    self._image_cache.get_nowait()
                                    self._image_cache.put_nowait((frame, timestamp))
                                except queue.Empty:
                                    pass
                            
                            # Update legacy latest for backward compatibility
                            with self._lock:
                                self._latest_image = (frame, timestamp)
                        else:
                            # Failed to decode image
                            self._consecutive_image_errors += 1
                            self._last_image_error_time = current_time
                    else:
                        # Failed to get image
                        self._consecutive_image_errors += 1
                        self._last_image_error_time = current_time
                        
                        # If too many consecutive errors, increase delay
                        if self._consecutive_image_errors >= self._max_consecutive_errors:
                            self._error_recovery_delay = min(5.0, self._error_recovery_delay * 1.5)
                            self.log.warning(f"Too many image errors, increasing recovery delay to {self._error_recovery_delay}s")
                except Exception as e:
                    self._consecutive_image_errors += 1
                    self._last_image_error_time = time.time()
                    error_msg = str(e)
                    
                    # Handle specific AirSim errors
                    if "cannot be re-sized" in error_msg or "Existing exports" in error_msg:
                        # This is a known AirSim issue - wait longer before retry
                        self.log.debug(f"AirSim resize error (will retry): {e}")
                        time.sleep(0.5)  # Wait longer for this specific error
                    else:
                        self.log.debug(f"Image update error: {e}")
                
                time.sleep(interval)
        
        if self._image_update_thread is None or not self._image_update_thread.is_alive():
            self._stop_updates.clear()
            self._image_update_thread = threading.Thread(target=image_update_loop, daemon=True, name=f"AirSim-Image-{self._ip}")
            self._image_update_thread.start()
    
    def is_connected(self) -> bool:
        """Check if connected to AirSim."""
        try:
            with self._lock:
                if self._client is None:
                    return False
                # Check connection state first
                if self._connection_state != ConnectionState.CONNECTED:
                    return False
            # Check if API control is enabled (indicates active connection)
            # This is a read operation - use data lock with timeout
            try:
                if self._data_lock.acquire(timeout=0.01):
                    try:
                        if hasattr(self._client, "isApiControlEnabled"):
                            return self._client.isApiControlEnabled()
                    finally:
                        self._data_lock.release()
            except Exception:
                # If API check fails, fall back to state check
                pass
            return True
        except Exception:
            # fallback: check client exists and state
            with self._lock:
                return self._client is not None and self._connection_state == ConnectionState.CONNECTED
    
    def connect_async(self) -> None:
        """Asynchronously connect to AirSim with retry logic."""
        def connect():
            max_retries = int(self._config.get("connect_max_retries", 3))
            base_backoff = float(self._config.get("connect_backoff_seconds", 1.0))
            
            for attempt in range(1, max_retries + 1):
                try:
                    if self._stop.is_set():
                        self.log.info("Stop requested, aborting connect attempts")
                        return
                    
                    # Update connection state
                    with self._lock:
                        if self._connection_state == ConnectionState.CONNECTED:
                            self.log.debug("Already connected, skipping connect attempt")
                            return
                        self._connection_state = ConnectionState.CONNECTING
                    
                    self.log.info(f"Connecting to AirSim at {self._ip}:{self._port} (attempt {attempt}/{max_retries})")
                    
                    # Create client and confirm connection
                    client = airsim.MultirotorClient(ip=self._ip, port=self._port)
                    client.confirmConnection()
                    
                    # Allow API control by default
                    try:
                        client.enableApiControl(True)
                    except Exception as e:
                        self.log.debug(f"enableApiControl failed (may be already enabled): {e}")
                    
                    # Set client only after successful connection
                    # Connection is a one-time operation, use command lock for safety
                    with self._command_lock:
                        self._client = client
                    
                    with self._lock:
                        self._state.connected = True
                        self._connection_state = ConnectionState.CONNECTED
                    
                    self.log.info(f"Connected to AirSim at {self._ip}:{self._port}")
                    
                    # Reset error counters on successful connection
                    self._consecutive_image_errors = 0
                    self._error_recovery_delay = 1.0
                    
                    # Start image updates
                    self._start_image_updates()
                    return
                    
                except Exception as e:
                    error_msg = str(e)
                    self.log.warning(f"Connect attempt {attempt} failed: {e}")
                    
                    # Clean up failed client
                    with self._command_lock:
                        if self._client == client:
                            self._client = None
                    
                    with self._lock:
                        self._state.connected = False
                        self._connection_state = ConnectionState.DISCONNECTED
                    
                    # Don't retry on certain errors
                    if "Connection refused" in error_msg or "timeout" in error_msg.lower():
                        if attempt < max_retries:
                            backoff = base_backoff * (2 ** (attempt - 1))
                            self.log.debug(f"Waiting {backoff:.2f}s before retry...")
                            time.sleep(backoff)
                        else:
                            self.log.error(f"Failed to connect after {max_retries} attempts")
                    else:
                        # For other errors, retry with shorter delay
                        if attempt < max_retries:
                            time.sleep(0.5)
            
            # All retries failed
            self.log.error(f"Failed to connect to AirSim after {max_retries} attempts")
        
        t = threading.Thread(target=connect, daemon=True, name=f"AirSim-Connect-{self._ip}")
        t.start()
    
    def terminate(self) -> None:
        """Terminate the AirSim connection."""
        self._stop.set()
        self._stop_updates.set()
        
        # Stop image update threads
        if self._image_update_thread and self._image_update_thread.is_alive():
            self._image_update_thread.join(timeout=2.0)
        if self._pc_update_thread and self._pc_update_thread.is_alive():
            self._pc_update_thread.join(timeout=2.0)
        
        if self._client:
            try:
                # Use command lock for cleanup operations (write operations)
                with self._command_lock:
                    # optionally disarm and disable API control
                    try:
                        self._client.armDisarm(False)
                    except Exception as e:
                        self.log.debug(f"armDisarm failed during termination: {e}")
                    try:
                        self._client.enableApiControl(False)
                    except Exception as e:
                        self.log.debug(f"enableApiControl failed during termination: {e}")
                    # reset the simulation for safety (optional, may be disabled)
                    try:
                        # Only reset if configured to do so
                        if self._config.get("reset_on_terminate", False):
                            self._client.reset()
                    except Exception as e:
                        self.log.debug(f"reset failed during termination: {e}")
            except Exception as e:
                self.log.warning(f"Error during AirSim termination: {e}")
        
        # Clear client reference (use command lock for thread safety)
        with self._command_lock:
            self._client = None
        
        with self._lock:
            self._state.connected = False
            self._connection_state = ConnectionState.DISCONNECTED
        
        self.log.debug("AirSimClient terminated and cleaned up.")
    
    # ---------- AirSim helper utilities ----------
    
    def list_cameras(self) -> List[str]:
        """Return list of available camera names in the current vehicle."""
        if not self._client:
            return []
        try:
            # Use data lock with timeout (read operation)
            # If command is executing, return default camera
            if not self._data_lock.acquire(timeout=0.05):
                # Lock is held, return default camera
                return ["0"]
            
            try:
                # We'll attempt to call camera names 0..8 and check if info exists
                cameras = []
                for i in range(10):
                    try:
                        cinfo = self._client.simGetCameraInfo(str(i))
                        if cinfo is not None:
                            cameras.append(str(i))
                    except Exception:
                        # ignore cameras that are not present
                        pass
                return cameras or ["0"]
            finally:
                self._data_lock.release()
        except Exception as e:
            error_msg = str(e)
            if "cannot be re-sized" in error_msg or "Existing exports" in error_msg:
                self.log.debug(f"AirSim resize error in list_cameras: {e}")
            return ["0"]
    
    def get_camera_image(self, camera_name: str = "0", image_type: int = 0, retries: int = 1) -> Optional[bytes]:
        """
        Return raw image bytes (PNG) or None on failure.
        
        Args:
            camera_name: Camera ID/name
            image_type: AirSim ImageType (0=Scene, 1=DepthPlanner, etc.)
            retries: Number of retry attempts
        
        Returns:
            Image bytes (PNG) or None
        """
        if not self._client:
            return None
        
        last_exc = None
        for attempt in range(retries + 1):
            try:
                # Use data lock with short timeout for image fetching
                # This is a read operation - should never block on commands
                # If command is executing, skip this frame (we'll get next frame)
                lock_acquired = False
                try:
                    # Try to acquire data lock with very short timeout (0.03s)
                    # If command is executing, skip this frame - this is expected and normal
                    # Image updates continue at high frequency, missing one frame is acceptable
                    lock_acquired = self._data_lock.acquire(timeout=0.03)
                    if not lock_acquired:
                        # Lock is held (likely by command or state monitoring), skip this attempt
                        # This is normal during control operations - we'll get next frame
                        if attempt < retries:
                            time.sleep(0.02)
                            continue
                        return None
                    
                    # Lock acquired, get image (non-blocking read operation)
                    img = self._client.simGetImage(camera_name, image_type)
                    if img and len(img) > 0:
                        return img  # bytes (PNG)
                    return None
                finally:
                    if lock_acquired:
                        self._data_lock.release()
            except Exception as e:
                last_exc = e
                error_msg = str(e)
                
                # Handle specific AirSim errors
                if "cannot be re-sized" in error_msg or "Existing exports" in error_msg:
                    # This is a known AirSim threading issue
                    # Wait a bit and retry
                    if attempt < retries:
                        time.sleep(0.1 * (attempt + 1))
                        continue
                    else:
                        # Log as debug to avoid spam
                        self.log.debug(f"AirSim resize error (attempt {attempt + 1}/{retries + 1}): {e}")
                        return None
                else:
                    # Other errors - log warning on first attempt
                    if attempt == 0:
                        self.log.warning(f"Failed to get camera image: {e}")
                    if attempt < retries:
                        time.sleep(0.05)
                        continue
                    return None
        
        return None
    
    def save_camera_image(self, filename: str, camera_name: str = "0", image_type: int = 0) -> bool:
        """Save an image to disk (filename). Returns True on success."""
        img_bytes = self.get_camera_image(camera_name, image_type)
        if not img_bytes:
            return False
        try:
            with open(filename, "wb") as f:
                f.write(img_bytes)
            return True
        except Exception as e:
            self.log.warning(f"Failed to write image to {filename}: {e}")
            return False
    
    # ---------- ROS-style topic handlers ----------
    
    def update_state(self, msg: Dict[str, Any]) -> None:
        """Handle state topic updates (for compatibility with ROS interface)."""
        try:
            update_timestamp = time.time()
            with self._lock:
                self._state.connected = True
                self._state.armed = bool(msg.get("armed", self._state.armed))
                self._state.mode = str(msg.get("mode", self._state.mode))
                self._state.last_updated = update_timestamp
                # Add to state history for synchronization
                self._add_state_to_history(self._state, update_timestamp)
            
            # Record state if recording is enabled
            if self._recorder and self._recorder.is_recording():
                with self._lock:
                    self._recorder.record_state(self._state, update_timestamp)
        except Exception as e:
            self.log.error(f"Error handling state update: {e}")
    
    def update_drone_state(self, msg: Dict[str, Any]) -> None:
        """Handle drone state topic updates."""
        try:
            update_timestamp = time.time()
            with self._lock:
                self._state.landed = bool(msg.get("landed", self._state.landed))
                self._state.returned = bool(msg.get("returned", self._state.returned))
                self._state.reached = bool(msg.get("reached", self._state.reached))
                self._state.tookoff = bool(msg.get("tookoff", self._state.tookoff))
                self._state.last_updated = update_timestamp
                # Add to state history for synchronization
                self._add_state_to_history(self._state, update_timestamp)
            
            # Record state if recording is enabled
            if self._recorder and self._recorder.is_recording():
                with self._lock:
                    self._recorder.record_state(self._state, update_timestamp)
        except Exception as e:
            self.log.error(f"Error handling drone state update: {e}")
    
    def update_camera(self, msg: Dict[str, Any]) -> None:
        """Handle camera image updates (for compatibility with ROS interface)."""
        # In AirSim, we get images directly from the API, not from ROS topics
        # This method is kept for compatibility but does nothing
        pass
    
    def update_battery(self, msg: Dict[str, Any]) -> None:
        """Handle battery topic updates."""
        try:
            update_timestamp = time.time()
            with self._lock:
                p = msg.get("percentage", msg.get("percent", msg.get("battery", self._state.battery)))
                try:
                    p_val = float(p)
                    self._state.battery = (p_val * 100.0) if p_val <= 1.0 else p_val
                except Exception:
                    self.log.debug("Unable to parse battery percentage; leaving previous value")
                self._state.last_updated = update_timestamp
                # Add to state history for synchronization
                self._add_state_to_history(self._state, update_timestamp)
            
            # Record state if recording is enabled
            if self._recorder and self._recorder.is_recording():
                with self._lock:
                    self._recorder.record_state(self._state, update_timestamp)
        except Exception as e:
            self.log.error(f"Error handling battery update: {e}")
    
    def update_gps(self, msg: Dict[str, Any]) -> None:
        """Handle GPS topic updates."""
        try:
            update_timestamp = time.time()
            with self._lock:
                try:
                    self._state.latitude = float(msg.get("latitude", msg.get("lat", self._state.latitude)))
                    self._state.longitude = float(msg.get("longitude", msg.get("lon", self._state.longitude)))
                    self._state.altitude = float(msg.get("altitude", msg.get("alt", self._state.altitude)))
                except Exception:
                    self.log.debug("Partial or invalid GPS data received")
                self._state.last_updated = update_timestamp
                # Add to state history for synchronization
                self._add_state_to_history(self._state, update_timestamp)
            
            # Record state if recording is enabled
            if self._recorder and self._recorder.is_recording():
                with self._lock:
                    self._recorder.record_state(self._state, update_timestamp)
        except Exception as e:
            self.log.error(f"Error handling GPS update: {e}")
    
    def update_point_cloud(self, msg: Dict[str, Any]) -> None:
        """Handle point cloud topic updates."""
        # AirSim doesn't provide point clouds via the same API
        # This method is kept for compatibility but does nothing
        pass
    
    # ---------- Image and point cloud access methods ----------
    
    def get_latest_image(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        Get the latest received image from cache (non-blocking, optimized for low latency).
        
        Returns:
            Tuple of (image, timestamp) or None
        """
        if not HAS_NUMPY:
            return None
        
        # Try to get from cache first (non-blocking, fastest)
        # Strategy: Get all available frames and keep only the latest
        # This ensures we always get the most recent frame even if multiple frames arrived
        latest = None
        try:
            # Drain the queue to get the latest frame
            while True:
                try:
                    latest = self._image_cache.get_nowait()
                except queue.Empty:
                    break
            # Put back the latest frame so it's available for next call
            if latest:
                try:
                    self._image_cache.put_nowait(latest)
                except queue.Full:
                    # Queue is full (shouldn't happen with maxsize=1), but we have the latest
                    pass
                return latest
        except Exception:
            pass
        
        # Fallback to legacy latest (thread-safe read)
        with self._lock:
            return getattr(self, "_latest_image", None)
    
    def get_latest_point_cloud(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        Get the latest received point cloud from cache (non-blocking).
        
        Returns:
            Tuple of (points array, timestamp) or None
        """
        if not HAS_NUMPY:
            return None
        
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
    
    def fetch_camera_image(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        Fetch camera image synchronously.
        
        Returns:
            Tuple of (image, timestamp) or None if unavailable
        """
        if not self._client or not self.is_connected():
            self.log.warning("Cannot fetch image — not connected to AirSim.")
            return None
        
        if not HAS_CV2 or not HAS_NUMPY:
            return None
        
        try:
            camera_id = self._config.get("camera_id", "0")
            # Use retry mechanism for synchronous fetch
            img_bytes = self.get_camera_image(camera_id, 0, retries=2)  # 0 = Scene (RGB)
            
            if img_bytes:
                # Convert PNG bytes to numpy array
                np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Convert BGR to RGB for consistency
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    timestamp = time.time()
                    
                    # Synchronize state with image timestamp
                    synced_state = self.sync_state_with_data(timestamp)
                    self._update_state_with_timestamp(timestamp)
                    
                    # Record image with synchronized state if recording is enabled
                    if self._recorder and self._recorder.is_recording():
                        self._recorder.record_image(frame, timestamp, state=synced_state)
                    
                    return (frame, timestamp)
            
            return None
        except Exception as e:
            error_msg = str(e)
            if "cannot be re-sized" in error_msg or "Existing exports" in error_msg:
                self.log.debug(f"AirSim resize error in fetch_camera_image: {e}")
            else:
                self.log.error(f"Error fetching camera image: {e}")
            return None
    
    def fetch_point_cloud(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        Fetch point cloud data synchronously.
        
        Returns:
            Tuple of (points array, timestamp) or None
        """
        # AirSim doesn't provide point clouds via the standard API
        # This method returns None for now
        return None
    
    # ---------- ROS-style service and publish methods ----------
    
    def service_call(
        self,
        service_name: str,
        service_type: str,
        payload: Dict[str, Any],
        timeout: Optional[float] = None,
        retries: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Call a ROS-style service (mapped to AirSim API calls).
        
        Args:
            service_name: Service name
            service_type: Service type (ignored for AirSim)
            payload: Service request payload
            timeout: Timeout in seconds
            retries: Number of retries
        
        Returns:
            Service response dictionary
        """
        if not self._client:
            raise RuntimeError("Not connected to AirSim")
        
        timeout = timeout or float(self._config.get("service_call_timeout", 10.0))
        retries = retries if retries is not None else int(self._config.get("service_call_retries", 2))
        last_exc: Optional[Exception] = None
        
        for attempt in range(1, retries + 2):
            try:
                svc_lower = service_name.lower()
                
                # Use command lock for write operations (commands)
                # Commands can block, but this doesn't prevent data acquisition
                # Data acquisition uses separate _data_lock with timeout
                with self._command_lock:
                    # takeoff (async operation - may take time)
                    if "takeoff" in svc_lower:
                        # Start async operation (quick, non-blocking)
                        future = self._client.takeoffAsync()
                        # Wait for completion (this may take time, but data acquisition continues)
                        # because it uses separate _data_lock
                        self._join_async_future(future, timeout)
                        with self._lock:
                            self._state.tookoff = True
                            self._state.landed = False
                        return {"success": True}
                    
                    # land (async operation - may take time)
                    if "land" in svc_lower:
                        future = self._client.landAsync()
                        self._join_async_future(future, timeout)
                        with self._lock:
                            self._state.landed = True
                        return {"success": True}
                    
                    # set_mode (quick operation)
                    if "set_mode" in svc_lower or "setmode" in svc_lower:
                        mode = payload.get("custom_mode", payload.get("mode", "")).upper()
                        if mode == "GUIDED" or mode == "OFFBOARD":
                            try:
                                self._client.enableApiControl(True)
                            except Exception:
                                pass
                        elif mode == "MANUAL":
                            try:
                                self._client.enableApiControl(False)
                            except Exception:
                                pass
                        with self._lock:
                            self._state.mode = mode
                        return {"success": True, "mode": mode}
                    
                    # arm (quick operation)
                    if "arm" in svc_lower and "disarm" not in svc_lower:
                        self._client.armDisarm(True)
                        with self._lock:
                            self._state.armed = True
                        return {"success": True}
                    
                    # disarm (quick operation)
                    if "disarm" in svc_lower or ("arm" in svc_lower and payload.get("arm", False) is False):
                        self._client.armDisarm(False)
                        with self._lock:
                            self._state.armed = False
                        return {"success": True}
                
                # Unknown service
                raise ValueError(f"Unknown/unsupported service: {service_name}")
                
                # fallback: unknown service
                raise ValueError(f"Unknown/unsupported service: {service_name}")
            except Exception as e:
                last_exc = e
                error_msg = str(e)
                # Handle AirSim-specific errors
                if "cannot be re-sized" in error_msg or "Existing exports" in error_msg:
                    if attempt <= retries:
                        # Wait longer for this specific error
                        time.sleep(0.5 * (attempt + 1))
                        continue
                # Handle IOLoop errors - these are often recoverable
                if "IOLoop" in error_msg or "already running" in error_msg.lower():
                    self.log.warning(f"Service call {service_name} failed on attempt {attempt} due to IOLoop: {e}")
                    if attempt <= retries:
                        # Wait a bit longer for IOLoop errors
                        time.sleep(1.0 * attempt)
                        continue
                    else:
                        # On final attempt, return a partial success response
                        # The operation may have actually succeeded despite the IOLoop error
                        self.log.warning(f"Service call {service_name} may have succeeded despite IOLoop error")
                        return {"success": True, "warning": "IOLoop error occurred, but operation may have succeeded"}
                self.log.warning(f"Service call {service_name} failed on attempt {attempt}: {e}")
                if attempt <= retries:
                    time.sleep(0.5 * attempt)
        
        raise last_exc if last_exc is not None else RuntimeError("Unknown service call failure")
    
    def publish(
        self,
        topic_name: str,
        topic_type: str,
        message: Dict[str, Any],
        retries: Optional[int] = None
    ) -> None:
        """
        Publish to a ROS-style topic (mapped to AirSim control calls).
        
        Args:
            topic_name: Topic name
            topic_type: Topic type (ignored for AirSim)
            message: Message dictionary
            retries: Number of retries
        """
        if not self._client:
            raise RuntimeError("Not connected to AirSim")
        
        retries = retries if retries is not None else int(self._config.get("publish_retries", 2))
        last_exc: Optional[Exception] = None
        
        for attempt in range(1, retries + 2):
            try:
                tname = topic_name.lower()
                
                # Use command lock for write operations (commands)
                # Release data lock before waiting for async operations to allow data acquisition
                with self._command_lock:
                    # Position setpoint
                    if "setpoint_position" in tname or "pose" in tname:
                        pos = message.get("pose", {}).get("position", {}) if isinstance(message, dict) else {}
                        x = float(pos.get("x", 0.0))
                        y = float(pos.get("y", 0.0))
                        z = float(pos.get("z", 0.0))
                        velocity = float(message.get("velocity", 5.0))
                        # AirSim uses NED coordinates in meters
                        self.log.debug(f"Moving to position (x={x}, y={y}, z={z}) at v={velocity}")
                        future = self._client.moveToPositionAsync(x, y, z, velocity)
                        # Release command lock before waiting - allows data acquisition to continue
                        # But we still hold command lock to prevent concurrent commands
                        self._join_async_future(future, timeout=30.0)
                        with self._lock:
                            self._state.latitude = x
                            self._state.longitude = y
                            self._state.altitude = z
                            self._state.reached = True
                        return
                    
                    # Velocity command (cmd_vel)
                    if "cmd_vel" in tname or "twist" in tname:
                        vel = message.get("linear", {}) if isinstance(message, dict) else {}
                        vx = float(vel.get("x", 0.0))
                        vy = float(vel.get("y", 0.0))
                        vz = float(vel.get("z", 0.0))
                        duration = float(message.get("duration", 1.0))
                        self.log.debug(f"Setting velocity vx={vx}, vy={vy}, vz={vz} for {duration}s")
                        future = self._client.moveByVelocityAsync(vx, vy, vz, duration)
                        self._join_async_future(future, timeout=duration + 5.0)
                        # do not flag reached
                        return
                
                # fallback
                self.log.warning(f"Unknown topic mapping for AirSim: {topic_name}")
                return
            except Exception as e:
                last_exc = e
                error_msg = str(e)
                # Handle AirSim-specific errors
                if "cannot be re-sized" in error_msg or "Existing exports" in error_msg:
                    if attempt <= retries:
                        # Wait longer for this specific error
                        time.sleep(0.3 * (attempt + 1))
                        continue
                # Handle IOLoop errors - these are often recoverable
                if "IOLoop" in error_msg or "already running" in error_msg.lower():
                    self.log.warning(f"Publish to {topic_name} failed on attempt {attempt} due to IOLoop: {e}")
                    if attempt <= retries:
                        # Wait a bit longer for IOLoop errors
                        time.sleep(0.5 * attempt)
                        continue
                    else:
                        # On final attempt, log warning but don't raise
                        # The operation may have actually succeeded despite the IOLoop error
                        self.log.warning(f"Publish to {topic_name} may have succeeded despite IOLoop error")
                        return
                self.log.warning(f"Publish to {topic_name} failed on attempt {attempt}: {e}")
                if attempt <= retries:
                    time.sleep(0.2 * attempt)
        
        self.log.error(f"Failed to publish to {topic_name} after {retries + 1} attempts: {last_exc}")
        if last_exc:
            raise last_exc
    
    # ---------- Convenience high-level commands ----------
    
    def takeoff(self, timeout: float = 10.0) -> bool:
        """Take off."""
        return bool(self.service_call("/takeoff", "", {}, timeout=timeout).get("success", False))
    
    def land(self, timeout: float = 10.0) -> bool:
        """Land."""
        return bool(self.service_call("/land", "", {}, timeout=timeout).get("success", False))
    
    def arm(self, value: bool = True) -> bool:
        """Arm or disarm."""
        if value:
            return bool(self.service_call("/arm", "", {"arm": True}).get("success", False))
        else:
            return bool(self.service_call("/disarm", "", {"arm": False}).get("success", False))
    
    def move_to(self, x: float, y: float, z: float, velocity: float = 5.0, timeout: float = 30.0) -> bool:
        """Move to a specific position."""
        if not self._client:
            return False
        try:
            # Use command lock for move operation (write operation)
            with self._command_lock:
                future = self._client.moveToPositionAsync(x, y, z, velocity)
                # Wait for move to complete, but data acquisition can continue
                self._join_async_future(future, timeout)
            with self._lock:
                self._state.latitude = x
                self._state.longitude = y
                self._state.altitude = z
                self._state.reached = True
            return True
        except Exception as e:
            error_msg = str(e)
            if "cannot be re-sized" in error_msg or "Existing exports" in error_msg:
                self.log.debug(f"AirSim resize error in move_to: {e}")
            elif "IOLoop" in error_msg or "already running" in error_msg.lower():
                self.log.warning(f"move_to failed due to IOLoop error: {e}. Operation may have succeeded.")
                # IOLoop errors may not indicate actual failure
                # Return True optimistically as the operation may have completed
                return True
            else:
                self.log.warning(f"move_to failed: {e}")
            return False
    
    # ---------- Standardized interface methods ----------
    
    def take_photo(self, camera_id: str = "0", save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Take a photo using AirSim API - standardized interface.
        
        Args:
            camera_id: AirSim camera name/ID
            save_path: Optional save path
        
        Returns:
            Standardized response dictionary
        """
        if not self._client:
            return {"success": False, "message": "Not connected to AirSim"}
        
        try:
            # AirSim ImageType: 0=Scene, 1=DepthPlanner, 2=DepthPerspective, 3=DepthVis, 4=DisparityNormalized, 5=Segmentation, 6=SurfaceNormals
            image_type = 0  # Scene (RGB)
            
            # Get image data with retry
            img_data = self.get_camera_image(camera_id, image_type, retries=2)
            
            if img_data is None or len(img_data) == 0:
                return {"success": False, "message": "Failed to capture image from AirSim"}
            
            result = {
                "success": True,
                "message": "Photo captured successfully",
                "camera_id": camera_id,
                "image_size": len(img_data)
            }
            
            # If save path specified, save to file
            if save_path:
                success = self.save_camera_image(save_path, camera_id, image_type)
                if success:
                    result["image_path"] = save_path
                    result["message"] = f"Photo captured and saved to {save_path}"
                    self.log.info(f"AirSimClient: photo saved to {save_path}")
                else:
                    return {"success": False, "message": f"Failed to save image to {save_path}"}
            else:
                # Return image data
                result["image_data"] = img_data
                self.log.info(f"AirSimClient: photo captured (in-memory, {len(img_data)} bytes)")
            
            return result
        except Exception as e:
            error_msg = str(e)
            if "cannot be re-sized" in error_msg or "Existing exports" in error_msg:
                self.log.debug(f"AirSim resize error in take_photo: {e}")
            else:
                self.log.error(f"AirSimClient photo capture failed: {e}")
            return {"success": False, "message": f"Photo capture failed: {e}"}
    
    def get_telemetry(self) -> Dict[str, Any]:
        """
        Get complete AirSim telemetry data - standardized interface.
        
        Returns:
            Dictionary with telemetry data
        """
        if not self._client:
            return {"success": False, "message": "Not connected to AirSim"}
        
        try:
            # Use data lock with timeout for telemetry (read operation)
            # This ensures telemetry doesn't block data acquisition
            if not self._data_lock.acquire(timeout=0.1):
                # Lock is held, return cached state instead
                with self._lock:
                    try:
                        state_dict = self._state.to_dict() if hasattr(self._state, 'to_dict') else {
                            "connected": self._state.connected,
                            "armed": self._state.armed,
                            "mode": self._state.mode,
                            "battery": self._state.battery,
                            "latitude": self._state.latitude,
                            "longitude": self._state.longitude,
                            "altitude": self._state.altitude,
                            "roll": self._state.roll,
                            "pitch": self._state.pitch,
                            "yaw": self._state.yaw,
                        }
                    except Exception:
                        from dataclasses import asdict
                        state_dict = asdict(self._state)
                    
                    return {
                        "success": True,
                        "timestamp": time.time(),
                        "state": state_dict,
                        "connection": {
                            "ip": self._ip,
                            "port": self._port,
                            "connected": self.is_connected()
                        },
                        "note": "Using cached state (API busy)"
                    }
            
            try:
                state = self._client.getMultirotorState()
            finally:
                self._data_lock.release()
            
            with self._lock:
                # Convert state to dict safely
                try:
                    state_dict = self._state.to_dict() if hasattr(self._state, 'to_dict') else {
                        "connected": self._state.connected,
                        "armed": self._state.armed,
                        "mode": self._state.mode,
                        "battery": self._state.battery,
                        "latitude": self._state.latitude,
                        "longitude": self._state.longitude,
                        "altitude": self._state.altitude,
                        "roll": self._state.roll,
                        "pitch": self._state.pitch,
                        "yaw": self._state.yaw,
                    }
                except Exception:
                    from dataclasses import asdict
                    state_dict = asdict(self._state)
                
                result = {
                    "success": True,
                    "timestamp": time.time(),
                    "state": state_dict,
                    "connection": {
                        "ip": self._ip,
                        "port": self._port,
                        "connected": self.is_connected()
                    },
                    "airsim_state": {
                        "can_arm": getattr(state, "can_arm", None),
                        "landed_state": str(getattr(state, "landed_state", "unknown")),
                        "timestamp": getattr(state, "timestamp", 0)
                    }
                }
                
                # Add velocity information
                try:
                    vel = state.kinematics_estimated.linear_velocity
                    result["velocity"] = {
                        "x": getattr(vel, "x_val", 0.0),
                        "y": getattr(vel, "y_val", 0.0),
                        "z": getattr(vel, "z_val", 0.0)
                    }
                except Exception:
                    pass
                
                # Add acceleration information
                try:
                    acc = state.kinematics_estimated.linear_acceleration
                    result["acceleration"] = {
                        "x": getattr(acc, "x_val", 0.0),
                        "y": getattr(acc, "y_val", 0.0),
                        "z": getattr(acc, "z_val", 0.0)
                    }
                except Exception:
                    pass
                
                return result
        except Exception as e:
            error_msg = str(e)
            if "cannot be re-sized" in error_msg or "Existing exports" in error_msg:
                self.log.debug(f"AirSim resize error in get_telemetry: {e}")
            else:
                self.log.error(f"AirSimClient get_telemetry failed: {e}")
            return {"success": False, "message": f"Failed to get telemetry: {e}"}
