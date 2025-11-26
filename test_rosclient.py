import json
import time
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt

from rosclient import RosClient, MockRosClient

# -----------------------------
# Logging 配置
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
)
logger = logging.getLogger("ROSBridgeDemo")


# -----------------------------
# 获取设备状态封装函数
# -----------------------------
def get_device_status(client, device_id: str, connection_url: str):
    """
    从 ROS 客户端中读取无人机状态并格式化为字典结构。
    捕获异常以防止连接失败导致程序中断。
    """
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


# -----------------------------
# 连接基础配置
# -----------------------------
connection_url = "ws://192.168.27.152:9090"
device_id = "drone1"

# 初始化 ROS 客户端
# client = MockRosClient(connection_url)
client = RosClient(connection_url)
# 连接客户端
client.connect_async()

logger.info("Connecting to ROSBridge...")

# -----------------------------
# 测试连接状态
# -----------------------------
is_conn = client.is_connected() if hasattr(client, "is_connected") else client.connected
logger.info(f"Connected: {is_conn}")

# -----------------------------
# 获取并打印无人机当前状态
# -----------------------------
status = get_device_status(client, device_id, connection_url)
logger.info(json.dumps(status, indent=2, ensure_ascii=False))

# -----------------------------
# 发布控制指令测试
# -----------------------------
control_topic = client._config.get("control_topic", "/control")
control_type = client._config.get("control_type", "controller_msgs/cmd")

for v in range(1, 4):
    # 发布控制命令
    client.publish(control_topic, control_type, {"cmd": v})

    # 发布目标点
    client.publish(
        "/goal_user2brig",
        "quadrotor_msgs/GoalSet",
        {"drone_id": v, "goal": [v, v * 10, v * 100]},
    )

    logger.info(f"Published control command: {v}")
    time.sleep(0.5)


# -----------------------------
# 获取相机画面
# -----------------------------
frame_data = client.fetch_camera_image()

if frame_data:
    frame, ts = frame_data
    logger.info(f"Image timestamp: {ts}")

    # 使用 Matplotlib 显示图像（转换 BGR → RGB）
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(f"Camera Frame (ts={ts:.3f})")
    plt.axis("off")
else:
    logger.warning("No camera frame received")


# -----------------------------
# 获取点云并可视化（最多显示 5000 个点）
# -----------------------------
cloud_data = client.fetch_point_cloud()

if cloud_data:
    points, ts = cloud_data
    num = len(points)
    logger.info(f"Point cloud: {num} points")

    if num > 0:
        pts = points[:min(num, 5000)]

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1)
        ax.set_title(f"Point Cloud Snapshot ({num} points)")
        plt.show()
else:
    logger.warning("No point cloud received")


# -----------------------------
# 结束连接
# -----------------------------
client.terminate()
logger.info("Disconnected.")
