#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import threading
import time
import re
import random
import signal
import sys
from urllib.parse import urlparse
from typing import Dict, Optional, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as FutureTimeoutError
import logging
import asyncio
import math

# third-party ROS client (kept as import; Mock used for tests)
import roslibpy

# SDK imports (kept as-is)
from device_protocol_sdk.abstract_device import AbstractDevice, ActionItem
from device_protocol_sdk.model.device_status import DeviceStatus, MessageLevel
from device_protocol_sdk.pusher import DevicePusher

# ==========================================================
# Logger Setup (single helper used everywhere)
# ==========================================================
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


# ==========================================================
# DroneState - thread-safe container (mutations guarded by RosClient)
# ==========================================================
class DroneState:
    def __init__(self) -> None:
        self.connected: bool = False
        self.armed: bool = False
        self.mode: str = ""
        self.battery: float = 100.0
        self.latitude: float = 0.0
        self.longitude: float = 0.0
        self.altitude: float = 0.0
        self.roll: float = 0.0
        self.pitch: float = 0.0
        self.yaw: float = 0.0
        self.landed: bool = True
        self.reached: bool = False
        self.returned: bool = False
        self.tookoff: bool = False
        self.last_updated: float = time.time()


# ==========================================================
# TopicServiceManager - thread-safe caches of topics & services
# ==========================================================
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
                    # not all roslibpy.Service implementations have unadvertise; guard it
                    unadvertise = getattr(s, "unadvertise", None)
                    if callable(unadvertise):
                        unadvertise()
                        self.log.info(f"Unadvertised service {k}")
                except Exception as e:
                    self.log.warning(f"Failed to unadvertise service {k}: {e}")
            self._topics.clear()
            self._services.clear()


# ==========================================================
# MockRosClient - for testing
# ==========================================================
class MockRosClient:
    def __init__(self, connection_str: str, config: Dict[str, Any] = None):
        self.connection_str = connection_str
        self._config = config or {}
        self._state = {
            "connected": True,
            "armed": False,
            "mode": "STANDBY",
            "battery": 100.0,
            "latitude": 22.5329,
            "longitude": 113.93029,
            "altitude": 0.0
        }
        self._lock = threading.RLock()
        self._terminated = False
        self.published_messages: List[Dict[str, Any]] = []
        self.service_calls: List[Dict[str, Any]] = []

    # Compatibility: is_connected() used by RosClient
    def is_connected(self) -> bool:
        with self._lock:
            return not self._terminated and bool(self._state.get("connected", False))

    def connect_async(self):
        with self._lock:
            self._state["connected"] = True

    def terminate(self):
        with self._lock:
            self._terminated = True
            self._state["connected"] = False

    def get_status(self) -> DeviceStatus:
        with self._lock:
            s = self._state
            return DeviceStatus(
                is_lock=0 if s["connected"] else 1,
                heartbeat=1 if s["connected"] else 0,
                battery=s["battery"],
                airspeed=0.0,
                groundspeed=0.0,
                yaw_degrees=0.0,
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
                lat=s["latitude"],
                lon=s["longitude"],
                alt=s["altitude"],
                vzspeed=0.0,
                height=s["altitude"]
            )

    def safe_service_call(self, service_name: str, service_type: str, payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        with self._lock:
            self.service_calls.append({
                "service_name": service_name,
                "service_type": service_type,
                "payload": payload
            })
            return {"mock": "success", "service": service_name}

    def safe_publish(self, topic_name: str, topic_type: str, message: Dict[str, Any], **kwargs):
        with self._lock:
            self.published_messages.append({
                "topic_name": topic_name,
                "topic_type": topic_type,
                "message": message
            })

    # state mutation helpers
    def set_mode(self, mode: str):
        with self._lock:
            self._state["mode"] = mode

    def set_armed(self, armed: bool):
        with self._lock:
            self._state["armed"] = armed

    def set_battery(self, percent: float):
        with self._lock:
            self._state["battery"] = percent

    def set_position(self, lat: float, lon: float, alt: float):
        with self._lock:
            self._state["latitude"] = lat
            self._state["longitude"] = lon
            self._state["altitude"] = alt


# ==========================================================
# RosClient - production wrapper around roslibpy.Ros
# ==========================================================
class RosClient:
    def __init__(self, connection_str: str, config: Optional[Dict[str, Any]] = None):
        self.connection_str = connection_str
        self._config = dict(config or {})
        self._lock = threading.RLock()
        self._ros: Optional[roslibpy.Ros] = None
        self._ts_mgr: Optional[TopicServiceManager] = None
        self._state = DroneState()
        self._stop = threading.Event()
        self.log = setup_logger(f"RosClient[{connection_str}]")
        self.log.setLevel(logging.INFO)

        # parse/validate websocket URL more robustly
        parsed = urlparse(connection_str)
        if parsed.scheme not in ("ws", "wss") or not parsed.hostname:
            raise ValueError(f"Invalid WebSocket URL: {connection_str}")
        self._host = parsed.hostname
        self._port = parsed.port or 9090
        self._path = parsed.path or "/"

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
            "connect_max_retries": 5,
            "connect_backoff_seconds": 1.0,
            "connect_backoff_max": 30.0,
            "service_call_timeout": 5.0,
            "service_call_retries": 2,
            "publish_retries": 2,
        }
        for k, v in defaults.items():
            self._config.setdefault(k, v)

        self._start_heartbeat()

    def _start_heartbeat(self, interval: float = 5.0) -> None:
        def loop():
            thread_name = threading.current_thread().name
            self.log.debug(f"Heartbeat thread started: {thread_name}")
            while not self._stop.is_set():
                try:
                    if not self.is_connected():
                        self.log.warning("Disconnected — attempting reconnect...")
                        try:
                            self.connect_async()
                        except Exception as e:
                            self.log.debug(f"connect_async failed inside heartbeat: {e}")
                    else:
                        self.log.debug("Heartbeat: connected")
                except Exception as exc:
                    self.log.exception(f"Heartbeat error: {exc}")
                finally:
                    slept = 0.0
                    while slept < interval and not self._stop.is_set():
                        time.sleep(0.2)
                        slept += 0.2

        t = threading.Thread(target=loop, daemon=True, name=f"Heartbeat-{self._host}:{self._port}")
        t.start()

    def is_connected(self) -> bool:
        with self._lock:
            if self._ros is None:
                return False
            # roslibpy.Ros exposes is_connected attribute (property)
            return bool(getattr(self._ros, "is_connected", False))

    def connect_async(self) -> None:
        def task():
            max_retries = int(self._config.get("connect_max_retries", 5))
            base_backoff = float(self._config.get("connect_backoff_seconds", 1.0))
            max_backoff = float(self._config.get("connect_backoff_max", 30.0))

            for attempt in range(1, max_retries + 1):
                if self._stop.is_set():
                    self.log.info("Stop requested — aborting connect attempts")
                    return
                if self.is_connected():
                    self.log.debug("Already connected (skipping connect attempt)")
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
                    self.log.warning(f"Connection attempt {attempt} failed: {e}", exc_info=True)

                backoff = min(max_backoff, base_backoff * (2 ** (attempt - 1)))
                jitter = random.uniform(0, backoff * 0.2)
                sleep_time = backoff + jitter
                self.log.debug(f"Sleeping {sleep_time:.2f}s before next connect attempt")
                time.sleep(sleep_time)

            self.log.critical("Failed to connect after maximum retries")

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
            # 新增里程计订阅
            self._ensure_ts_mgr().topic(conf["odom_topic"], conf["odom_type"]).subscribe(self.update_odom)
            # state subscribe
            self._ensure_ts_mgr().topic(conf["drone_state_topic"], conf['drone_state_topic']).subscribe(self.update_drone_state)
        except Exception as e:
            self.log.exception(f"Failed to subscribe topics: {e}")

    def update_odom(self, msg: Dict[str, Any]) -> None:
        try:
            with self._lock:
                # 获取四元数
                q = msg.get("pose", {}).get("pose", {}).get("orientation", {})
                x = float(q.get("x", 0.0))
                y = float(q.get("y", 0.0))
                z = float(q.get("z", 0.0))
                w = float(q.get("w", 1.0))

                # 四元数转欧拉角 (roll, pitch, yaw)
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
                
                roll = math.degrees(roll)
                pitch = math.degrees(pitch)
                yaw = math.degrees(yaw)

                # 更新状态
                self._state.roll = roll
                self._state.pitch = pitch
                self._state.yaw = yaw

                # 可选：更新位置
                pos = msg.get("pose", {}).get("pose", {}).get("position", {})
                self._state.latitude = float(pos.get("x", self._state.latitude))
                self._state.longitude = float(pos.get("y", self._state.longitude))
                self._state.altitude = float(pos.get("z", self._state.altitude))

                self._state.last_updated = time.time()
            self.log.debug(f"Odometry updated: roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}")
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
            self.log.debug(f"State updated: landed={self._state.landed}, returned={self._state.returned}")
        except Exception as e:
            self.log.exception(f"Error handling state update: {e}")    

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

    def get_status(self) -> DeviceStatus:
        with self._lock:
            s = self._state
            return DeviceStatus(
                is_lock=0 if s.connected else 1,
                heartbeat=1 if s.connected else 0,
                battery=s.battery,
                airspeed=0.0,
                groundspeed=0.0,
                yaw_degrees=0.0,
                roll=s.roll,
                pitch=s.pitch,
                yaw=s.yaw,
                lat=s.latitude,
                lon=s.longitude,
                alt=s.altitude,
                vzspeed=0.0,
                height=s.altitude,
                landed=s.landed,
                returned=s.returned,
                reached=s.reached,
                tookoff=s.tookoff
            )


# ==========================================================
# BaseAction and concrete actions
# ==========================================================
class BaseAction:
    name: str = ""
    command_type: str = ""
    description: str = ""
    schema: Dict[str, Any] = {}

    def __init__(self) -> None:
        self.log = setup_logger(self.__class__.__name__)

    def validate(self, params: Dict[str, Any]) -> Optional[str]:
        required = self.schema.get("required", [])
        for r in required:
            if r not in params:
                return f"Missing required parameter: {r}"
        return None

    def execute(self, client: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Action must implement execute()")


class TakeoffAction(BaseAction):
    name = "Takeoff"
    command_type = "takeoff"
    description = "Command the vehicle to take off to a target altitude"
    schema = {"type": "object", "properties": {"altitude": {"type": "number"}}, "required": ["altitude"]}

    def execute(self, client: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        self.log.info(f"Executing TAKEOFF with params: {params}")
        try:
            altitude = float(params["altitude"])
            payload = {"altitude": altitude}
            resp = client.safe_service_call(client._config["takeoff_service"], client._config["takeoff_type"], payload)
            self.log.info(f"Takeoff service response: {resp}")
            return {"status": "success", "response": resp}
        except Exception as e:
            self.log.exception("Takeoff failed")
            return {"status": "error", "message": str(e)}


class LandAction(BaseAction):
    name = "Land"
    command_type = "land"
    description = "Command the vehicle to land at current position"

    def execute(self, client: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        self.log.info("Executing LAND command")
        try:
            resp = client.safe_service_call(client._config["land_service"], client._config["land_type"], {})
            self.log.info(f"Land service response: {resp}")
            return {"status": "success", "response": resp}
        except Exception as e:
            self.log.exception("Land failed")
            return {"status": "error", "message": str(e)}


class MoveAction(BaseAction):
    name = "Move to Position"
    command_type = "move_to_position"
    description = "Publish a setpoint position message to make vehicle move"
    schema = {
        "type": "object",
        "properties": {"latitude": {"type": "number"}, "longitude": {"type": "number"}, "altitude": {"type": "number"}},
        "required": ["latitude", "longitude"]
    }

    def execute(self, client: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        self.log.info(f"Executing MOVE command: {params}")
        try:
            lat = float(params["latitude"])
            lon = float(params["longitude"])
            alt = float(params.get("altitude", 10.0))
            msg = {"pose": {"position": {"x": lon, "y": lat, "z": alt}}}
            client.safe_publish(client._config["move_topic"], client._config["move_topic_type"], msg)
            self.log.info(f"Move command sent to ({lat}, {lon}, alt={alt})")
            return {"status": "success"}
        except Exception as e:
            self.log.exception("Move failed")
            return {"status": "error", "message": str(e)}


class SetModeAction(BaseAction):
    name = "Set Mode"
    command_type = "set_mode"
    description = "Set vehicle mode via set_mode service"
    schema = {"type": "object", "properties": {"mode": {"type": "string"}}, "required": ["mode"]}

    def execute(self, client: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        self.log.info(f"Executing SET_MODE with params: {params}")
        try:
            custom_mode = str(params["mode"])
            payload = {"base_mode": 0, "custom_mode": custom_mode}
            resp = client.safe_service_call(client._config["set_mode_service"], client._config["set_mode_type"], payload)
            self.log.info(f"Set mode response: {resp}")
            return {"status": "success", "response": resp}
        except Exception as e:
            self.log.exception("SetMode failed")
            return {"status": "error", "message": str(e)}
        

class GoalAction(BaseAction):
    name = "Goal Command"
    command_type = "goal"
    description = "Send a goal position to the drone using /goal_user2brig topic"
    schema = {
        "type": "object",
        "properties": {
            "drone_id": {"type": "integer"},
            "goal": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3
            }
        },
        "required": ["drone_id", "goal"]
    }

    def execute(self, client: RosClient, params: Dict[str, Any]) -> Dict[str, Any]:
        self.log.info(f"Executing GOAL with params: {params}")
        try:
            drone_id = int(params["drone_id"])
            goal = params["goal"]
            if len(goal) != 3:
                raise ValueError("Goal must be a 3-element list [x, y, z]")

            topic_name = client._config.get("goal_topic", "/goal_user2brig")
            topic_type = client._config.get("goal_type", "quadrotor_msgs/GoalSet")

            msg = {
                "drone_id": drone_id,
                "goal": [float(goal[0]), float(goal[1]), float(goal[2])]
            }

            client.safe_publish(topic_name, topic_type, msg)
            self.log.info(f"Published goal {msg} to {topic_name}")
            return {"status": "success", "message": f"Goal sent to drone {drone_id}"}

        except Exception as e:
            self.log.exception("Goal publish failed")
            return {"status": "error", "message": str(e)}


class ControlAction(BaseAction):
    name = "Control Command"
    command_type = "control"
    description = "Publish a control integer to configured control topic 起飞/control=1，降落/control=2，返航/control=3，急停/control=5，航点飞行/control=4且/goal_with_id发布id（默认为1）和坐标"
    schema = {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}

    def execute(self, client: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        self.log.info(f"Executing CONTROL with params: {params}")
        try:
            val = int(params["value"])
            topic_name = client._config.get("control_topic", "/control")
            topic_type = client._config.get("control_type", "controller_msgs/cmd")
            msg = {"cmd": val}
            print(msg)
            client.safe_publish(topic_name, topic_type, msg)
            self.log.info(f"Published control value={val} to {topic_name}")
            return {"status": "success", "message": f"Control {val} sent"}
        except Exception as e:
            self.log.exception("Control publish failed")
            return {"status": "error", "message": str(e)}


# ==========================================================
# ActionHandler - registry and dispatcher for actions
# ==========================================================
class ActionHandler:
    def __init__(self):
        self._actions: Dict[str, BaseAction] = {}
        self.log = setup_logger("ActionHandler")

    def register(self, act: BaseAction) -> None:
        self._actions[act.command_type] = act
        self.log.debug(f"Registered action: {act.command_type}")

    def list_actions(self) -> List[ActionItem]:
        return [
            ActionItem(name=a.name, command_type=a.command_type, description=a.description, params=a.schema)
            for a in self._actions.values()
        ]

    def dispatch(self, client: Any, cmd: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self.log.info(f"Dispatching action '{cmd}' with params={params}")
        action = self._actions.get(cmd)
        if not action:
            self.log.warning(f"Unknown action: {cmd}")
            return {"status": "error", "message": f"Unknown command: {cmd}"}
        err = action.validate(params or {})
        if err:
            self.log.warning(f"Parameter validation failed: {err}")
            return {"status": "error", "message": err}
        return action.execute(client, params or {})


# ==========================================================
# ROSBridgeDroneDevice - device implementation
# ==========================================================
class ROSBridgeDroneDevice(AbstractDevice):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._config = config or {}
        self._clients: Dict[str, Any] = {}
        self._clients_lock = threading.RLock()
        self._actions = ActionHandler()
        self.log = setup_logger("ROSBridgeDroneDevice")

        for act in [GoalAction(), ControlAction()]:
            self._actions.register(act)
        self.log.info("ROSBridgeDroneDevice initialized.")

        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except Exception:
            self.log.debug("Signal handlers not installed (embedding environment?)")

    @property
    def protocol_name(self) -> str:
        return "ros_drone"

    def _create_client(self, device_id: str, connection_str: str) -> Any:
        """
        Create and cache a client (Mock or real) for a given connection string.
        Returns the client instance, or raises on failure.
        """
        with self._clients_lock:
            client = self._clients.get(connection_str)
            if client:
                return client

            self.log.info(f"Creating new ROS client for {connection_str}")
            merged_conf = dict(self._config)
            use_mock = bool(merged_conf.get("use_mock_client", True))
            try:
                if use_mock:
                    client = MockRosClient(connection_str, merged_conf)
                else:
                    client = RosClient(connection_str, merged_conf)
                # Start connection (non-blocking)
                # both MockRosClient and RosClient implement connect_async()
                try:
                    client.connect_async()
                except Exception as e:
                    self.log.warning(f"Client.connect_async failed for {connection_str}: {e}", exc_info=True)
                self._clients[connection_str] = client
                return True, client
            except Exception as e:
                self.log.exception(f"Device connection failed: {connection_str} - {e}")
                raise False

    def _close_client(self, connection_str: str) -> bool:
        self.log.info(f"Closing client {connection_str}")
        with self._clients_lock:
            client = self._clients.pop(connection_str, None)
        if not client:
            self.log.debug(f"No client found for {connection_str}")
            return True
        try:
            terminate_fn = getattr(client, "terminate", None)
            if callable(terminate_fn):
                terminate_fn()
            return True
        except Exception as e:
            self.log.warning(f"Error terminating client {connection_str}: {e}", exc_info=True)
            return False

    def get_device_status(self, client, device_id: str, connection_str: str) -> DeviceStatus:
        """
        Return DeviceStatus for a given connection_str.
        The 'client' arg is ignored (kept for compatibility with SDK signature).
        """
        try:
            with self._clients_lock:
                client_obj = self._clients.get(connection_str)
            if not client_obj:
                self.log.warning(f"No client for {connection_str}, creating one...")
                try:
                    client_obj = self._create_client(device_id, connection_str)
                except Exception as e:
                    self.log.error(f"Failed to create client for status: {e}", exc_info=True)
                    # return offline status
                    return DeviceStatus(
                        is_lock=1,
                        heartbeat=0,
                        battery=0.0,
                        airspeed=0.0,
                        groundspeed=0.0,
                        yaw_degrees=0.0,
                        roll=0.0,
                        pitch=0.0,
                        yaw=0.0,
                        lat=0.0,
                        lon=0.0,
                        alt=0.0,
                        vzspeed=0.0,
                        height=0.0
                    )

            status = client_obj.get_status()
            self.log.info(f"Status for {connection_str}: {status}")
            return status
        except Exception as e:
            self.log.error(f"Failed to get device status for {connection_str}: {e}", exc_info=True)
            return DeviceStatus(
                is_lock=1,
                heartbeat=0,
                battery=0.0,
                airspeed=0.0,
                groundspeed=0.0,
                yaw_degrees=0.0,
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
                lat=0.0,
                lon=0.0,
                alt=0.0,
                vzspeed=0.0,
                height=0.0
            )

    def get_action_list(self) -> List[ActionItem]:
        return self._actions.list_actions()

    def execute(self, client, device_id: str, connection_str: str, command_type: str, params: Dict[str, Any]):
        """
        Execute an action on the client associated with connection_str.
        If client does not exist yet, create it.
        """
        try:
            self.log.info(f"Execute command '{command_type}' for {connection_str}")
            with self._clients_lock:
                client_obj = self._clients.get(connection_str)
            if not client_obj:
                # create client; errors bubble to outer except and returned as error result
                client_obj = self._create_client(device_id, connection_str)

            result = self._actions.dispatch(client_obj, command_type, params or {})
            return result
        except TimeoutError:
            self.log.error(f"Command execution timed out for {connection_str}: {command_type}")
            return {"status": "error", "message": "Command execution timed out"}
        except Exception as e:
            self.log.error(f"Command execution failed for {connection_str}, command: {command_type}, error: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def shutdown(self) -> None:
        self.log.info("Shutting down ROSBridgeDroneDevice - closing all clients")
        with self._clients_lock:
            keys = list(self._clients.keys())
        for k in keys:
            try:
                ok = self._close_client(k)
                if not ok:
                    self.log.warning(f"Client {k} closed with errors")
            except Exception as e:
                self.log.warning(f"Error closing client {k}: {e}", exc_info=True)

    def _signal_handler(self, signum, frame) -> None:
        self.log.info(f"Received signal {signum}, shutting down device.")
        try:
            self.shutdown()
        finally:
            sys.exit(0)


# ==========================================================
# Async main - runs DevicePusher
# ==========================================================
async def main():
    main_logger = setup_logger("Main")
    try:
        async with DevicePusher(lambda: ROSBridgeDroneDevice()) as pusher:
            server_address = "192.168.209.166:50058"
            main_logger.info(f"Connecting to server {server_address}")
            await pusher.connect_server(server_address, 'device_description')
            main_logger.info(f"Connected to server {server_address}")
            main_logger.info("Device server started, waiting for commands")
            await asyncio.Future()
    except asyncio.CancelledError:
        main_logger.info("Main task cancelled, shutting down cleanly")
    except KeyboardInterrupt:
        main_logger.info("Program interrupted by user (KeyboardInterrupt)")
    except Exception as e:
        main_logger.exception(f"Program failed: {e}")
        raise
    finally:
        main_logger.info("Main exiting.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.exception(f"Fatal error in main: {e}")
        sys.exit(1)
