from __future__ import annotations

import base64
import math
import random
import threading
import time
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List

import cv2
import numpy as np
from urllib.parse import urlparse

# third-party ROS client (kept as import; Mock used for tests)
import roslibpy


# ======================
# Logger Setup
# ======================
def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(threadName)s] [%(name)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


logger = setup_logger("ROSBridgeDroneDevice", level=logging.INFO)


# ======================
# DroneState (dataclass)
# Thread safety: container is mutated only while holding lock in client classes
# ======================
@dataclass
class DroneState:
    connected: bool = False
    armed: bool = False
    mode: str = ""
    battery: float = 100.0
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    landed: bool = True
    reached: bool = False
    returned: bool = False
    tookoff: bool = False
    last_updated: float = field(default_factory=time.time)


# ======================
# TopicServiceManager (unchanged but cleaned)
# ======================
class TopicServiceManager:
    def __init__(self, ros: roslibpy.Ros, conn_id: str, logger_level=logging.DEBUG):
        self._ros = ros
        self._topics: Dict[str, roslibpy.Topic] = {}
        self._services: Dict[str, roslibpy.Service] = {}
        self._lock = threading.Lock()
        self.log = setup_logger(f"TopicService[{conn_id}]")
        self.log.setLevel(logger_level)

    def _key(self, name: str, t: str) -> str:
        return f"{name}:{t}"

    def topic(self, name: str, ttype: str) -> roslibpy.Topic:
        key = self._key(name, ttype)
        with self._lock:
            if key not in self._topics:
                self._topics[key] = roslibpy.Topic(self._ros, name, ttype)
                self.log.debug(f"Created topic {key}")
            return self._topics[key]

    def service(self, name: str, stype: str) -> roslibpy.Service:
        key = self._key(name, stype)
        with self._lock:
            if key not in self._services:
                self._services[key] = roslibpy.Service(self._ros, name, stype)
                self.log.debug(f"Created service {key}")
            return self._services[key]

    def close_all(self) -> None:
        with self._lock:
            for k, t in list(self._topics.items()):
                try:
                    t.unsubscribe()
                    self.log.info(f"Unsubscribed topic {k}")
                except Exception as e:
                    self.log.warning(f"Failed to unsubscribe topic {k}: {e}")
            for k, s in list(self._services.items()):
                try:
                    unadvertise = getattr(s, "unadvertise", None)
                    if callable(unadvertise):
                        unadvertise()
                        self.log.info(f"Unadvertised service {k}")
                except Exception as e:
                    self.log.warning(f"Failed to unadvertise service {k}: {e}")
            self._topics.clear()
            self._services.clear()


# ======================
# Helper: retry/backoff
# ======================
def _exponential_backoff(base: float, attempt: int, max_backoff: float, jitter_fraction: float = 0.2) -> float:
    backoff = min(max_backoff, base * (2 ** (attempt - 1)))
    jitter = random.uniform(0, backoff * jitter_fraction)
    return backoff + jitter


# ======================
# Abstract base class for Ros clients
# ======================
class RosClientBase(ABC):
    """
    Abstract base for Ros clients. Defines the unified API that both MockRosClient and
    real RosClient implement.
    """

    def __init__(self, connection_str: str, config: Optional[Dict[str, Any]] = None):
        self.connection_str = connection_str
        self._config = dict(config or {})
        self._lock = threading.RLock()
        self._state = DroneState()
        self._ts_mgr: Optional[TopicServiceManager] = None
        self._stop = threading.Event()
        self.log = setup_logger(f"RosClientBase[{connection_str}]")
        self.log.setLevel(logging.INFO)

    # resource management
    def __enter__(self) -> "RosClientBase":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.terminate()
        except Exception:
            pass

    # lifecycle
    @abstractmethod
    def is_connected(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def connect_async(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def terminate(self) -> None:
        raise NotImplementedError

    # publish / service
    @abstractmethod
    def safe_publish(self, topic_name: str, topic_type: str, message: Dict[str, Any], **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def safe_service_call(self, service_name: str, service_type: str, payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    # state accessors
    def get_status(self) -> DroneState:
        with self._lock:
            # return a shallow copy to avoid accidental external mutation
            s = DroneState(**self._state.__dict__)
            return s

    def get_position(self) -> Tuple[float, float, float]:
        with self._lock:
            return (self._state.latitude, self._state.longitude, self._state.altitude)

    def get_orientation(self) -> Tuple[float, float, float]:
        with self._lock:
            return (self._state.roll, self._state.pitch, self._state.yaw)

    def get_battery(self) -> float:
        with self._lock:
            return self._state.battery

    def is_armed(self) -> bool:
        with self._lock:
            return self._state.armed

    def get_flight_mode(self) -> str:
        with self._lock:
            return self._state.mode

    def is_landed(self) -> bool:
        with self._lock:
            return self._state.landed

    def has_reached_goal(self) -> bool:
        with self._lock:
            return self._state.reached

    def has_returned_home(self) -> bool:
        with self._lock:
            return self._state.returned

    def has_taken_off(self) -> bool:
        with self._lock:
            return self._state.tookoff


# ======================
# MockRosClient (for tests)
# ======================
class MockRosClient(RosClientBase):
    def __init__(self, connection_str: str, config: Optional[Dict[str, Any]] = None):
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

        self.log = setup_logger(f"MockRosClient[{connection_str}]")
        self.log.setLevel(logging.DEBUG)

    def is_connected(self) -> bool:
        with self._lock:
            return not getattr(self, "_terminated", False) and bool(self._state.connected)

    def connect_async(self) -> None:
        with self._lock:
            # immediate connect in mock
            self._terminated = False
            self._state.connected = True
            self.log.debug("Mock: connected (connect_async)")

    def terminate(self) -> None:
        with self._lock:
            self._terminated = True
            self._state.connected = False
            self.log.debug("Mock: terminated")

    def safe_service_call(self, service_name: str, service_type: str, payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
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

    def safe_publish(self, topic_name: str, topic_type: str, message: Dict[str, Any], **kwargs) -> None:
        with self._lock:
            self.published_messages.append({
                "topic_name": topic_name,
                "topic_type": topic_type,
                "message": message
            })
            self.log.debug(f"Mock publish recorded: {topic_name}")

    # state mutation helpers for tests
    def set_mode(self, mode: str):
        with self._lock:
            self._state.mode = mode
            self._state.last_updated = time.time()

    def set_armed(self, armed: bool):
        with self._lock:
            self._state.armed = armed
            self._state.last_updated = time.time()

    def set_battery(self, percent: float):
        with self._lock:
            self._state.battery = percent
            self._state.last_updated = time.time()

    def set_position(self, lat: float, lon: float, alt: float):
        with self._lock:
            self._state.latitude = lat
            self._state.longitude = lon
            self._state.altitude = alt
            self._state.last_updated = time.time()


# ======================
# RosClient - production wrapper around roslibpy.Ros
# ======================
class RosClient(RosClientBase):
    def __init__(self, connection_str: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(connection_str, config=config)

        # default config values (safe defaults)
        defaults = {
            "state_topic": "/mavros/state",
            "state_type": "mavros_msgs/State",
            "odom_topic": "/mavros/local_position/odom",
            "odom_type": "nav_msgs/Odometry",
            "battery_topic": "/mavros/battery",
            "battery_type": "sensor_msgs/BatteryState",
            "gps_topic": "/mavros/global_position/global",
            "gps_type": "sensor_msgs/NavSatFix",
            "takeoff_service": "/mavros/cmd/takeoff",
            "takeoff_type": "mavros_msgs/CommandTOL",
            "land_service": "/mavros/cmd/land",
            "land_type": "mavros_msgs/CommandTOL",
            "set_mode_service": "/mavros/set_mode",
            "set_mode_type": "mavros_msgs/SetMode",
            "move_topic": "/mavros/setpoint_position/global",
            "move_topic_type": "geometry_msgs/PoseStamped",
            "control_topic": "/control",
            "control_type": "controller_msgs/cmd",
            "drone_state_topic": "drone_state",
            "drone_state_type": "controller_msgs/DroneState",
            "goal_topic": "/goal_user2brig",
            "goal_type": "quadrotor_msgs/GoalSet",
            "camera_topic": "/camera/image_raw/compressed",
            "camera_type": "sensor_msgs/CompressedImage",
            "point_cloud_topic": "/drone_1_cloud_registered",
            "point_cloud_type": "sensor_msgs/PointCloud2",
            "connect_max_retries": 5,
            "connect_backoff_seconds": 1.0,
            "connect_backoff_max": 30.0,
            "service_call_timeout": 5.0,
            "service_call_retries": 2,
            "publish_retries": 2,
            "heartbeat_interval": 5.0,
            "logger_level": logging.INFO,
        }
        for k, v in defaults.items():
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
        
        self._latest_image: Optional[Tuple[np.ndarray, float]] = None
        self._latest_point_cloud: Optional[Tuple[np.ndarray, float]] = None
        
        self._connecting = False  # 标志正在连接
        self._connect_lock = threading.Lock()
        self._stop.clear()

        self.log = setup_logger(f"RosClient[{self._host}:{self._port}]")
        self.log.setLevel(self._config.get("logger_level", logging.INFO))

        # start heartbeat thread
        self._start_heartbeat(float(self._config.get("heartbeat_interval", 5.0)))

    def _start_heartbeat(self, interval: float):
        def loop():
            self.log.debug("Heartbeat thread started")
            while not self._stop.is_set():
                try:
                    if not self.is_connected() and not self._connecting:
                        self.log.info("Heartbeat detected disconnected — attempting reconnect")
                        self.connect_async()
                except Exception as e:
                    self.log.exception(f"Heartbeat error: {e}")
                finally:
                    slept = 0.0
                    while slept < interval and not self._stop.is_set():
                        time.sleep(0.2)
                        slept += 0.2

        t = threading.Thread(target=loop, daemon=True, name=f"Heartbeat-{self._host}:{self._port}")
        t.start()

    def is_connected(self) -> bool:
        with self._lock:
            return bool(self._ros and getattr(self._ros, "is_connected", False))


    def connect_async(self):
        """异步连接，如果已有连接线程在运行则跳过"""
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
                    if self.is_connected():
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

                    sleep_time = _exponential_backoff(base_backoff, attempt, max_backoff)
                    self.log.debug(f"Sleeping {sleep_time:.2f}s before next connect attempt")
                    time.sleep(sleep_time)

                self.log.critical("Failed to connect after maximum retries")
            finally:
                with self._connect_lock:
                    self._connecting = False

        t = threading.Thread(target=task, daemon=True, name=f"Connect-{self._host}:{self._port}")
        t.start()

    def terminate(self) -> None:
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
        self.log.debug("RosClient terminated and cleaned up.")

    def _subscribe_topics(self) -> None:
        if not self.is_connected():
            self.log.warning("Cannot subscribe — not connected.")
            return
        conf = self._config
        try:
            self.log.info("Subscribing to state/battery/gps topics.")
            self._ensure_ts_mgr().topic(conf["state_topic"], conf["state_type"]).subscribe(self.update_state)
            self._ensure_ts_mgr().topic(conf["battery_topic"], conf["battery_type"]).subscribe(self.update_battery)
            self._ensure_ts_mgr().topic(conf["gps_topic"], conf["gps_type"]).subscribe(self.update_gps)
            # odometry subscription
            self._ensure_ts_mgr().topic(conf["odom_topic"], conf["odom_type"]).subscribe(self.update_odom)
            # drone state
            self._ensure_ts_mgr().topic(conf["drone_state_topic"], conf["drone_state_type"]).subscribe(self.update_drone_state)
            # camera and point cloud
            self._ensure_ts_mgr().topic(conf["camera_topic"], conf["camera_type"]).subscribe(self.update_camera)
            self._ensure_ts_mgr().topic(conf["point_cloud_topic"], conf["point_cloud_type"]).subscribe(self.update_point_cloud)
        except Exception as e:
            self.log.exception(f"Failed to subscribe topics: {e}")

    # ---------- topic handlers ----------
    def update_odom(self, msg: Dict[str, Any]) -> None:
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

    def update_state(self, msg: Dict[str, Any]) -> None:
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
            with self._lock:
                frame = None
                # CompressedImage: typically has 'data' with base64 or raw bytes (roslibpy may already give bytes)
                if "data" in msg and isinstance(msg["data"], (bytes, bytearray)):
                    # already bytes
                    np_arr = np.frombuffer(msg["data"], np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                elif "data" in msg and isinstance(msg["data"], str):
                    # base64-encoded string (common in JSON-wrapped messages)
                    img_data = base64.b64decode(msg["data"])
                    np_arr = np.frombuffer(img_data, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                elif "encoding" in msg and "data" in msg:
                    # raw image with explicit height/width
                    height = int(msg.get("height", 0))
                    width = int(msg.get("width", 0))
                    encoding = msg.get("encoding", "bgr8")
                    channels = 3 if encoding in ("rgb8", "bgr8") else 1
                    img_data = np.frombuffer(msg["data"], dtype=np.uint8)
                    if height > 0 and width > 0 and img_data.size == height * width * channels:
                        frame = img_data.reshape((height, width, channels))
                    else:
                        self.log.debug("Raw image shape mismatch; cannot reshape")
                else:
                    self.log.warning("Received unknown camera message format")
                    return

                if frame is None:
                    self.log.warning("Failed to decode camera frame")
                    return

                timestamp = time.time()
                self._latest_image = (frame, timestamp)
                self._state.last_updated = timestamp

            self.log.debug(f"Received camera frame at {timestamp:.3f}s")
        except Exception as e:
            self.log.exception(f"Error decoding camera image: {e}")

    def fetch_camera_image(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        Pull one image from camera topic synchronously (no persistent subscription required).
        Returns (image, timestamp) or None if unavailable.
        """
        try:
            with self._lock:
                if not self.is_connected():
                    self.log.warning("Cannot fetch image — not connected to ROS.")
                    return None

                conf = self._config
                topic_name = conf.get("camera_topic", "/camera/image_raw/compressed")
                topic_type = conf.get("camera_type", "sensor_msgs/CompressedImage")

                topic = self._ensure_ts_mgr().topic(topic_name, topic_type)

            # call get_message outside lock (rospy/roslibpy blocking operations shouldn't hold client lock)
            msg = topic.get_message(timeout=3.0)
            if not msg:
                self.log.warning("No image message received from ROS.")
                return None

            # decode as in update_camera
            frame = None
            if "data" in msg and isinstance(msg["data"], (bytes, bytearray)):
                np_arr = np.frombuffer(msg["data"], np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            elif "data" in msg and isinstance(msg["data"], str):
                img_data = base64.b64decode(msg["data"])
                np_arr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
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
        with self._lock:
            return getattr(self, "_latest_image", None)

    # ---------- Point cloud handling ----------
    def update_point_cloud(self, msg: Dict[str, Any]) -> None:
        try:
            result = self._decode_point_cloud(msg)
            if result:
                points, ts = result
                with self._lock:
                    self._latest_point_cloud = (points, ts)
                    self._state.last_updated = ts
                self.log.debug(f"Received point cloud frame with {len(points)} points at {ts:.3f}s")
        except Exception as e:
            self.log.exception(f"Error handling point cloud update: {e}")

    def _decode_point_cloud(self, msg: Dict[str, Any]) -> Optional[Tuple[np.ndarray, float]]:
        """Common decoder for PointCloud2-like messages. Returns (points, timestamp) or None."""
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

    def fetch_point_cloud(self) -> Optional[Tuple[np.ndarray, float]]:
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

    # ---------- battery/gps ----------
    def update_battery(self, msg: Dict[str, Any]) -> None:
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

    # ---------- utility ----------
    def _ensure_ts_mgr(self) -> TopicServiceManager:
        with self._lock:
            if not self._ts_mgr and self._ros:
                self._ts_mgr = TopicServiceManager(self._ros, f"{self._host}:{self._port}")
            if not self._ts_mgr:
                raise RuntimeError("TopicServiceManager is not available (not connected)")
            return self._ts_mgr

    def safe_service_call(self, service_name: str, service_type: str, payload: Dict[str, Any],
                          timeout: Optional[float] = None, retries: Optional[int] = None) -> Dict[str, Any]:
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
                self.log.warning(last_exc)
            except Exception as e:
                last_exc = e
                self.log.warning(f"Service call {service_name} failed on attempt {attempt}: {e}")
            if attempt <= retries:
                backoff = min(5.0, 0.5 * (2 ** (attempt - 1)))
                jitter = random.uniform(0, backoff * 0.2)
                time.sleep(backoff + jitter)

        raise last_exc if last_exc is not None else RuntimeError("Unknown service call failure")

    def safe_publish(self, topic_name: str, topic_type: str, message: Dict[str, Any],
                     retries: Optional[int] = None) -> None:
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
