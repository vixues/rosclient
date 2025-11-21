import json
import logging
import time
import cv2
from typing import Any, Dict
from rosclient import RosClient, MockRosClient


# --------------------------------------------------------
# Logging Configuration
# --------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
)
logger = logging.getLogger("ROSBridgeTest")


# --------------------------------------------------------
# Utility Function: Retrieve Drone Status
# --------------------------------------------------------
def get_device_status(client: RosClient, device_id: str, connection_url: str) -> Dict[str, Any]:
    """Extract and return key status information from the drone via RosClient."""
    try:
        state = client.get_status()
        pos = client.get_position()
        ori = client.get_orientation()

        return {
            "device_id": device_id,
            "connection": connection_url,
            "connected": state.connected,
            "armed": state.armed,
            "mode": state.mode,
            "battery": f"{state.battery:.1f}%",
            "position": {"lat": pos[0], "lon": pos[1], "alt": pos[2]},
            "orientation": {"roll": ori[0], "pitch": ori[1], "yaw": ori[2]},
            "landed": state.landed,
            "tookoff": state.tookoff,
            "reached": state.reached,
            "returned": state.returned,
            "last_update": time.strftime("%H:%M:%S", time.localtime(state.last_updated))
        }
    except Exception as e:
        logger.error(f"Failed to retrieve device status: {e}")
        return {"error": str(e)}


def test_connection(client: RosClient, connection_url: str) -> bool:
    """Check ROSBridge connection status with retry."""
    logger.info("Testing ROSBridge connection...")

    try:
        if hasattr(client, "is_connected") and callable(client.is_connected):
            connected = client.is_connected()
        else:
            connected = getattr(client, "connected", False)

        if connected:
            logger.info("ROSBridge connected successfully.")
            return True
    except Exception as e:
        logger.warning(f"Connection check failed: {e}")

    time.sleep(5)
    logger.error(f"Unable to connect to ROSBridge: {connection_url}")
    return False


def test_device_status(client: RosClient, device_id: str, connection_url: str):
    logger.info("Testing device status retrieval...")
    status = get_device_status(client, device_id, connection_url)
    logger.info(json.dumps(status, indent=2, ensure_ascii=False))
    return status


def test_control_publish(client: RosClient):
    logger.info("Testing control topic publishing...")

    control_topic = client._config.get("control_topic", "/control")
    control_type = client._config.get("control_type", "controller_msgs/cmd")

    for val in range(1, 4):
        try:
            client.publish(control_topic, control_type, {"cmd": val})
            client.publish(
                '/goal_user2brig',
                "quadrotor_msgs/GoalSet",
                {'drone_id': int(val), 'goal': [val, val * 10, val * 100]}
            )
            logger.info(f"Successfully published control command: {val}")
        except Exception as e:
            logger.error(f"Failed to publish control command {val}: {e}")
        time.sleep(1)


def test_status_updates(client: RosClient, device_id: str, connection_url: str):
    logger.info("Starting periodic status update test...")
    for i in range(3):
        time.sleep(2)
        status = get_device_status(client, device_id, connection_url)
        logger.info(f"Status update [{i + 1}]: {json.dumps(status, indent=2, ensure_ascii=False)}")


def test_camera_snapshot(client: RosClient):
    logger.info("Testing instant camera frame capture...")

    try:
        frame_data = client.fetch_camera_image()
        if frame_data:
            frame, ts = frame_data
            logger.info(f"Successfully retrieved camera frame, timestamp: {ts:.3f}")
            cv2.imshow("Drone Camera Snapshot", frame)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
        else:
            logger.warning("No camera frame received")
    except Exception as e:
        logger.warning(f"Failed to fetch camera frame: {e}")


def test_point_cloud_snapshot(client: RosClient):
    logger.info("Testing instant point cloud capture...")

    try:
        cloud_data = client.fetch_point_cloud()
        if cloud_data:
            points, ts = cloud_data
            num_points = len(points)
            logger.info(f"Retrieved point cloud: {num_points} points, timestamp: {ts:.3f}")

            if num_points > 0:
                import matplotlib.pyplot as plt
                subset = points[:min(5000, num_points)]
                fig = plt.figure("Drone Point Cloud Snapshot")
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2], s=1)
                ax.set_title(f"Drone Point Cloud Snapshot ({num_points} points)")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                plt.show(block=False)
                plt.pause(3.0)
                plt.close(fig)
        else:
            logger.warning("No point cloud data received")
    except Exception as e:
        logger.warning(f"Failed to fetch point cloud data: {e}")


def test_disconnect(client: RosClient):
    logger.info("Closing connection...")
    client.terminate()
    time.sleep(1)
    logger.info("Connection closed safely")


def main(Mock=False):
    connection_url = "ws://192.168.27.152:9090"
    device_id = "drone1"

    if Mock:
        client = MockRosClient(connection_url)
    else:
        try:
            logger.info(f"Connecting to ROSBridge server at {connection_url} ...")
            client = RosClient(connection_url)
            logger.info("Connection initiated successfully.")
        except Exception as e:
            logger.error(f"Failed to start ROSBridge connection: {e}")
            return

    # Connection Test
    connected = test_connection(client, connection_url)
    if not connected:
        logger.error("Could not establish a ROSBridge connection. Exiting tests.")
        return

    time.sleep(10)
    test_device_status(client, device_id, connection_url)
    test_control_publish(client)
    test_status_updates(client, device_id, connection_url)
    # test_camera_snapshot(client)
    # test_point_cloud_snapshot(client)
    test_disconnect(client)


if __name__ == "__main__":
    try:
        main(Mock=True)
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.exception(f"Unexpected error during testing: {e}")