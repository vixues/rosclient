#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import threading
import signal
import sys
from typing import Dict, Optional, Any, List
import logging
import asyncio

# third-party ROS client - use rosclient package
from rosclient import RosClient, MockRosClient
from rosclient.models.drone import DroneState as RosClientDroneState

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
# Helper function to convert DroneState to DeviceStatus
# ==========================================================
def drone_state_to_device_status(state: RosClientDroneState) -> DeviceStatus:
    """
    Convert rosclient DroneState to SDK DeviceStatus.
    
    Args:
        state: DroneState from rosclient
        
    Returns:
        DeviceStatus for SDK
    """
    return DeviceStatus(
        is_lock=0 if state.connected else 1,
        heartbeat=1 if state.connected else 0,
        battery=state.battery,
        airspeed=0.0,
        groundspeed=0.0,
        yaw_degrees=0.0,
        roll=state.roll,
        pitch=state.pitch,
        yaw=state.yaw,
        lat=state.latitude,
        lon=state.longitude,
        alt=state.altitude,
        vzspeed=0.0,
        height=state.altitude,
        landed=state.landed,
        returned=state.returned,
        reached=state.reached,
        tookoff=state.tookoff
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
    description = (
        "Command the vehicle to take off to a specified target altitude. "
        "This action calls the takeoff service which initiates the takeoff sequence. "
        "The vehicle will ascend vertically to the specified altitude in meters. "
        "The vehicle must be armed and in an appropriate flight mode (e.g., GUIDED) before takeoff. "
        "Parameters: altitude (number) - the target altitude in meters above ground level. "
        "Example: {\"altitude\": 10.0} commands the vehicle to take off to 10 meters altitude. "
        "Note: Ensure the vehicle is properly armed and the flight mode supports takeoff operations."
    )
    schema = {
        "type": "object",
        "properties": {
            "altitude": {
                "type": "number",
                "description": "Target altitude in meters above ground level",
                "minimum": 0
            }
        },
        "required": ["altitude"]
    }

    def execute(self, client: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        self.log.info(f"Executing TAKEOFF with params: {params}")
        try:
            altitude = float(params["altitude"])
            payload = {"altitude": altitude}
            resp = client.service_call(client._config["takeoff_service"], client._config["takeoff_type"], payload)
            self.log.info(f"Takeoff service response: {resp}")
            return {"status": "success", "response": resp}
        except Exception as e:
            self.log.exception("Takeoff failed")
            return {"status": "error", "message": str(e)}


class LandAction(BaseAction):
    name = "Land"
    command_type = "land"
    description = (
        "Command the vehicle to land at its current position. "
        "This action calls the land service which initiates the landing sequence. "
        "The vehicle will descend vertically and land at the current location. "
        "The landing procedure is automatic and the vehicle will maintain its current horizontal position. "
        "This command does not require any parameters. "
        "Example: {} - empty parameters object triggers landing. "
        "Note: The vehicle should be in a safe altitude and location before executing the land command. "
        "Ensure there are no obstacles in the landing path."
    )

    def execute(self, client: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        self.log.info("Executing LAND command")
        try:
            resp = client.service_call(client._config["land_service"], client._config["land_type"], {})
            self.log.info(f"Land service response: {resp}")
            return {"status": "success", "response": resp}
        except Exception as e:
            self.log.exception("Land failed")
            return {"status": "error", "message": str(e)}


class MoveAction(BaseAction):
    name = "Move to Position"
    command_type = "move_to_position"
    description = (
        "Publish a setpoint position message to command the vehicle to move to a specific location. "
        "This action publishes a position setpoint to the configured move topic, which the vehicle's "
        "flight controller will use to navigate to the target position. "
        "The coordinates can be in local or global frame depending on the vehicle's configuration. "
        "Parameters: latitude (number, required) - target latitude or X coordinate, "
        "longitude (number, required) - target longitude or Y coordinate, "
        "altitude (number, optional) - target altitude or Z coordinate in meters (default: 10.0). "
        "Example: {\"latitude\": 22.5329, \"longitude\": 113.93029, \"altitude\": 15.0} "
        "commands the vehicle to move to the specified coordinates at 15 meters altitude. "
        "Note: The vehicle must be in a flight mode that accepts position setpoints (e.g., GUIDED, OFFBOARD)."
    )
    schema = {
        "type": "object",
        "properties": {
            "latitude": {
                "type": "number",
                "description": "Target latitude or X coordinate"
            },
            "longitude": {
                "type": "number",
                "description": "Target longitude or Y coordinate"
            },
            "altitude": {
                "type": "number",
                "description": "Target altitude or Z coordinate in meters (default: 10.0)",
                "default": 10.0
            }
        },
        "required": ["latitude", "longitude"]
    }

    def execute(self, client: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        self.log.info(f"Executing MOVE command: {params}")
        try:
            lat = float(params["latitude"])
            lon = float(params["longitude"])
            alt = float(params.get("altitude", 10.0))
            msg = {"pose": {"position": {"x": lon, "y": lat, "z": alt}}}
            client.publish(client._config["move_topic"], client._config["move_topic_type"], msg)
            self.log.info(f"Move command sent to ({lat}, {lon}, alt={alt})")
            return {"status": "success"}
        except Exception as e:
            self.log.exception("Move failed")
            return {"status": "error", "message": str(e)}


class SetModeAction(BaseAction):
    name = "Set Mode"
    command_type = "set_mode"
    description = (
        "Set the vehicle's flight mode via the set_mode service. "
        "This action allows you to change the vehicle's operational mode, which determines "
        "how the vehicle responds to commands and controls. "
        "Common flight modes include: MANUAL, STABILIZE, ALT_HOLD, POSITION, GUIDED, AUTO, RTL (Return to Launch), etc. "
        "The exact available modes depend on the vehicle type and autopilot configuration. "
        "Parameters: mode (string, required) - the flight mode name to set. "
        "Example: {\"mode\": \"GUIDED\"} sets the vehicle to GUIDED mode. "
        "Example: {\"mode\": \"RTL\"} sets the vehicle to Return-to-Launch mode. "
        "Note: Some mode changes may require the vehicle to be in a specific state (e.g., disarmed for certain modes). "
        "Ensure the requested mode is supported by your vehicle's autopilot."
    )
    schema = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "description": "Flight mode name (e.g., GUIDED, AUTO, RTL, MANUAL, etc.)"
            }
        },
        "required": ["mode"]
    }

    def execute(self, client: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        self.log.info(f"Executing SET_MODE with params: {params}")
        try:
            custom_mode = str(params["mode"])
            payload = {"base_mode": 0, "custom_mode": custom_mode}
            resp = client.service_call(client._config["set_mode_service"], client._config["set_mode_type"], payload)
            self.log.info(f"Set mode response: {resp}")
            return {"status": "success", "response": resp}
        except Exception as e:
            self.log.exception("SetMode failed")
            return {"status": "error", "message": str(e)}
        

class GoalAction(BaseAction):
    name = "Goal Command"
    command_type = "goal"
    description = (
        "Send a goal position command to a specific drone using the /goal_user2brig topic. "
        "This command publishes a goal waypoint (x, y, z coordinates) to the specified drone ID. "
        "The goal coordinates are typically in the local or global frame depending on the drone's configuration. "
        "The drone will navigate to the specified position after receiving this command. "
        "Parameters: drone_id (integer) - the unique identifier of the target drone, "
        "goal (array of 3 numbers) - the target position [x, y, z] in meters. "
        "Example: {\"drone_id\": 1, \"goal\": [10.0, 20.0, 5.0]} sends drone 1 to position (10, 20, 5)."
    )
    schema = {
        "type": "object",
        "properties": {
            "drone_id": {
                "type": "integer",
                "description": "Unique identifier of the target drone",
                "minimum": 1
            },
            "goal": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3,
                "description": "Target position coordinates [x, y, z] in meters"
            }
        },
        "required": ["drone_id", "goal"]
    }

    def execute(self, client: Any, params: Dict[str, Any]) -> Dict[str, Any]:
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

            client.publish(topic_name, topic_type, msg)
            self.log.info(f"Published goal {msg} to {topic_name}")
            return {"status": "success", "message": f"Goal sent to drone {drone_id}"}

        except Exception as e:
            self.log.exception("Goal publish failed")
            return {"status": "error", "message": str(e)}


class ControlAction(BaseAction):
    name = "Control Command"
    command_type = "control"
    description = (
        "Publish a control command integer to the configured control topic. "
        "This action sends a simple integer command to control the vehicle's basic operations. "
        "The control values have the following meanings: "
        "1 = Takeoff (起飞) - Command the vehicle to take off, "
        "2 = Land (降落) - Command the vehicle to land, "
        "3 = Return to Launch (返航) - Command the vehicle to return to its launch/home position, "
        "4 = Waypoint Flight (航点飞行) - Enable waypoint flight mode (requires goal command with drone_id and coordinates), "
        "5 = Emergency Stop (急停) - Immediately stop all vehicle movement and hover in place. "
        "Parameters: value (integer, required) - the control command value (1-5). "
        "Example: {\"value\": 1} commands the vehicle to take off. "
        "Example: {\"value\": 2} commands the vehicle to land. "
        "Example: {\"value\": 3} commands the vehicle to return to launch. "
        "Example: {\"value\": 5} triggers emergency stop. "
        "Note: For waypoint flight (value=4), you must also publish a goal command with drone_id and coordinates "
        "to the /goal_with_id topic. The default drone_id is 1 if not specified."
    )
    schema = {
        "type": "object",
        "properties": {
            "value": {
                "type": "integer",
                "description": "Control command value: 1=Takeoff, 2=Land, 3=Return to Launch, 4=Waypoint Flight, 5=Emergency Stop",
                "enum": [1, 2, 3, 4, 5],
                "minimum": 1,
                "maximum": 5
            }
        },
        "required": ["value"]
    }

    def execute(self, client: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        self.log.info(f"Executing CONTROL with params: {params}")
        try:
            val = int(params["value"])
            topic_name = client._config.get("control_topic", "/control")
            topic_type = client._config.get("control_type", "controller_msgs/cmd")
            msg = {"cmd": val}
            print(msg)
            client.publish(topic_name, topic_type, msg)
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
            use_mock = bool(merged_conf.get("use_mock_client", False))
            
            # Map old config keys to rosclient format if needed
            # rosclient uses DEFAULT_TOPICS, but we can override via config
            # Goal and Control topics
            merged_conf.setdefault("goal_topic", "/goal_user2brig")
            merged_conf.setdefault("goal_type", "quadrotor_msgs/GoalSet")
            merged_conf.setdefault("control_topic", "/control")
            merged_conf.setdefault("control_type", "controller_msgs/cmd")
            # Service configurations (for TakeoffAction, LandAction, SetModeAction)
            merged_conf.setdefault("takeoff_service", "/mavros/cmd/takeoff")
            merged_conf.setdefault("takeoff_type", "mavros_msgs/CommandTOL")
            merged_conf.setdefault("land_service", "/mavros/cmd/land")
            merged_conf.setdefault("land_type", "mavros_msgs/CommandTOL")
            merged_conf.setdefault("set_mode_service", "/mavros/set_mode")
            merged_conf.setdefault("set_mode_type", "mavros_msgs/SetMode")
            # Move topic configuration
            merged_conf.setdefault("move_topic", "/mavros/setpoint_position/global")
            merged_conf.setdefault("move_topic_type", "geometry_msgs/PoseStamped")
            
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
                return client
            except Exception as e:
                self.log.exception(f"Device connection failed: {connection_str} - {e}")
                raise

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

            drone_state = client_obj.get_status()
            device_status = drone_state_to_device_status(drone_state)
            self.log.info(f"Status for {connection_str}: {device_status}")
            return device_status
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
