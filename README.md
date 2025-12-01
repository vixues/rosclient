# ROS Client

一个用于连接和控制ROS（Robot Operating System）设备的Python客户端库，特别针对无人机控制场景进行了优化。

## 功能特性

- **WebSocket连接**: 通过rosbridge连接ROS设备
- **状态监控**: 实时获取无人机状态（位置、姿态、电池等）
- **图像处理**: 接收和显示相机图像
- **点云处理**: 接收和可视化3D点云数据
- **控制命令**: 发送ROS Topic控制命令
- **自动重连**: 支持连接断开自动重连机制
- **Mock模式**: 提供Mock客户端用于测试
- **GUI工具**: 图形界面测试工具

## 安装

### 基本安装

```bash
# 克隆仓库
git clone <repository-url>
cd rosclient

# 安装依赖
pip install roslibpy numpy
```

### 完整安装（包含图像和点云功能）

```bash
pip install roslibpy numpy opencv-python Pillow matplotlib
```

## 快速开始

### 基本使用

```python
from rosclient import RosClient

# 创建客户端
client = RosClient("ws://localhost:9090")

# 异步连接
client.connect_async()

# 等待连接
import time
time.sleep(2)

# 获取状态
if client.is_connected():
    state = client.get_status()
    print(f"模式: {state.mode}, 电池: {state.battery}%")
    
    position = client.get_position()
    print(f"位置: {position}")

# 发送控制命令
client.publish("/control", "controller_msgs/cmd", {"cmd": 1})

# 断开连接
client.terminate()
```

### 使用Mock客户端（测试）

```python
from rosclient import MockRosClient

# 创建Mock客户端（无需实际ROS连接）
client = MockRosClient("ws://localhost:9090")
client.connect_async()

# 使用方式与真实客户端相同
state = client.get_status()
print(f"状态: {state}")
```

## 项目结构

```text
rosclient/
├── rosclient/              # 主包
│   ├── core/              # 核心功能
│   │   ├── base.py        # 基类
│   │   └── topic_service_manager.py  # Topic/Service管理器
│   ├── clients/           # 客户端实现
│   │   ├── ros_client.py  # 生产环境客户端
│   │   ├── mock_client.py # Mock客户端
│   │   └── config.py      # 配置
│   ├── models/            # 数据模型
│   │   ├── drone.py       # 无人机状态模型
│   │   └── state.py       # 连接状态枚举
│   └── utils/             # 工具函数
│       ├── logger.py      # 日志工具
│       └── backoff.py     # 指数退避算法
├── tests/                 # 测试套件
├── rosclient_gui_test.py  # GUI测试工具
└── README.md
```

## 主要API

### RosClient

```python
# 连接管理
client.connect_async()      # 异步连接
client.terminate()          # 断开连接
client.is_connected()       # 检查连接状态

# 状态获取
client.get_status()         # 获取完整状态
client.get_position()       # 获取位置 (lat, lon, alt)
client.get_orientation()    # 获取姿态 (roll, pitch, yaw)

# 图像和点云
client.fetch_camera_image()        # 获取相机图像
client.get_latest_image()          # 获取最新图像
client.fetch_point_cloud()        # 获取点云数据
client.get_latest_point_cloud()    # 获取最新点云

# 消息发布
client.publish(topic, type, message)  # 安全发布消息

# 服务调用
client.service_call(service, type, payload)  # 安全调用服务
```

### 数据模型

```python
from rosclient import DroneState, ConnectionState

# DroneState包含所有无人机状态信息
state = DroneState(
    connected=True,
    armed=False,
    mode="GUIDED",
    battery=85.5,
    latitude=22.5329,
    longitude=113.93029,
    altitude=100.0,
    # ... 更多字段
)

# ConnectionState枚举
state = ConnectionState.CONNECTED
```

## GUI测试工具

项目包含一个图形界面测试工具，方便测试和调试：

```bash
python rosclient_gui_test.py
```

GUI工具提供以下功能：

- 连接配置和测试
- 实时状态监控
- 图像显示
- 点云可视化
- 控制命令发送
- 网络测试

