# ROS Client 开发文档

## 目录

- [概述](#概述)
- [架构设计](#架构设计)
- [模块说明](#模块说明)
- [核心类和方法](#核心类和方法)
- [使用示例](#使用示例)
- [扩展指南](#扩展指南)
- [API 参考](#api-参考)

---

## 概述

`rosclient` 是一个用于连接和操作 ROS (Robot Operating System) 的 Python 客户端库，主要用于无人机控制和数据采集。该库提供了完整的 ROS 通信功能，包括主题订阅、服务调用、数据记录和回放等功能。

### 主要特性

- **多客户端支持**: 支持真实 ROS 连接、模拟客户端、rosbag 回放和 AirSim 集成
- **高性能数据采集**: 异步数据采集，支持图像、点云和状态数据的同步记录
- **智能数据处理**: 内置图像处理器和点云处理器，支持插件化算法扩展
- **数据记录与回放**: 完整的数据记录和回放系统，支持时间同步
- **线程安全**: 所有操作都是线程安全的，支持多线程环境
- **易于扩展**: 清晰的架构设计，便于二次开发和功能扩展

---

## 架构设计

### 整体架构

```
rosclient/
├── clients/          # 客户端实现层
│   ├── ros_client.py      # 真实 ROS 客户端
│   ├── mock_client.py      # 模拟客户端（用于测试）
│   ├── rosbag_client.py   # rosbag 回放客户端
│   └── airsim_client.py   # AirSim 仿真客户端
├── core/            # 核心功能层
│   ├── base.py            # 基础客户端类（抽象基类）
│   ├── topic_service_manager.py  # 主题和服务管理器
│   ├── recorder.py        # 数据记录器
│   └── player.py          # 数据回放器
├── models/          # 数据模型层
│   ├── drone.py           # 无人机状态模型
│   └── state.py            # 连接状态枚举
├── processors/     # 数据处理器层
│   ├── image_processor.py  # 图像处理器
│   ├── pointcloud_processor.py  # 点云处理器
│   └── plugins.py         # 算法插件（YOLO, SAM3等）
└── utils/          # 工具函数层
    ├── logger.py          # 日志工具
    └── backoff.py         # 重试策略
```

### 架构层次

```
┌─────────────────────────────────────────┐
│          Application Layer              │  (GUI, 业务逻辑)
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│          Client Layer                   │  (RosClient, MockRosClient)
│  - 连接管理                              │
│  - 数据订阅/发布                         │
│  - 状态管理                              │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│          Core Layer                     │  (RosClientBase, Recorder, Player)
│  - 生命周期管理                           │
│  - 数据记录/回放                         │
│  - 状态同步                              │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│          Processor Layer                │  (ImageProcessor, PointCloudProcessor)
│  - 数据解码                              │
│  - 数据转换                              │
│  - 算法处理                              │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│          Model Layer                    │  (DroneState, RosTopic)
│  - 数据模型定义                           │
│  - 状态管理                              │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│          ROS Bridge                     │  (roslibpy, rosbridge)
└─────────────────────────────────────────┘
```

### 设计模式

1. **模板方法模式**: `RosClientBase` 定义了客户端的基本框架，子类实现具体细节
2. **策略模式**: 不同的客户端实现（RosClient, MockRosClient）可以互换使用
3. **观察者模式**: 通过回调函数处理主题消息
4. **工厂模式**: `TopicServiceManager` 管理主题和服务的创建
5. **插件模式**: 图像处理器支持算法插件扩展

### 数据流

#### 订阅数据流

```
ROS Topic
    ↓
roslibpy (WebSocket)
    ↓
TopicServiceManager
    ↓
RosClient (消息处理)
    ↓
Processor (数据解码)
    ↓
Cache/State Update
    ↓
Application (使用数据)
```

#### 记录数据流

```
ROS Topic
    ↓
RosClient (接收消息)
    ↓
Processor (解码数据)
    ↓
State Synchronization (状态同步)
    ↓
Recorder (异步记录)
    ↓
Queue (缓冲)
    ↓
Background Thread (写入)
    ↓
File (保存)
```

#### 回放数据流

```
File (记录文件)
    ↓
Recorder.load() (加载)
    ↓
RecordPlayer (回放控制)
    ↓
Time Synchronization (时间同步)
    ↓
Callback (回调函数)
    ↓
Application (使用数据)
```

---

## 模块说明

### 1. clients/ - 客户端实现层

#### 1.1 RosClient (`ros_client.py`)

真实 ROS 客户端，通过 WebSocket 连接到 rosbridge。

**主要功能**:
- 异步连接到 ROS
- 订阅/发布主题
- 调用 ROS 服务
- 处理图像、点云、状态等数据

**关键特性**:
- 自动重连机制（指数退避）
- 高频率数据缓存（队列机制，保留最新帧）
- 线程安全的状态管理
- 支持数据记录

**使用示例**:
```python
from rosclient import RosClient

client = RosClient("ws://localhost:9090")
client.connect_async()

# 等待连接
import time
while not client.is_connected():
    time.sleep(0.1)

# 获取最新图像
image, timestamp = client.get_latest_image()
```

#### 1.2 MockRosClient (`mock_client.py`)

模拟客户端，用于测试和开发，无需实际 ROS 环境。

**主要功能**:
- 模拟 ROS 连接和数据
- 支持从文件加载真实数据
- 支持回放记录文件
- 生成模拟图像和点云

**配置选项**:
```python
config = {
    "playback_file": "path/to/recording.rosrec",  # 回放文件
    "playback_speed": 1.0,                        # 回放速度
    "playback_loop": True,                        # 是否循环
    "real_image_path": "path/to/images/",         # 真实图像路径
    "real_pointcloud_path": "path/to/pointclouds/",  # 真实点云路径
    "image_update_interval": 0.033,               # 图像更新间隔（秒）
    "pointcloud_update_interval": 0.033          # 点云更新间隔（秒）
}

client = MockRosClient("mock://test", config=config)
```

#### 1.3 RosbagClient (`rosbag_client.py`)

rosbag 文件回放客户端，用于回放已录制的 rosbag 文件。

#### 1.4 AirSimClient (`airsim_client.py`)

AirSim 仿真环境客户端，用于连接 Microsoft AirSim 仿真器。

### 2. core/ - 核心功能层

#### 2.1 RosClientBase (`base.py`)

所有客户端的抽象基类，定义了客户端的基本接口和通用功能。

**核心方法**:

- `connect_async()`: 异步连接（抽象方法，子类实现）
- `terminate()`: 终止连接（抽象方法，子类实现）
- `get_status()`: 获取当前状态
- `get_position()`: 获取位置
- `get_orientation()`: 获取姿态
- `update_odom()`: 更新里程计数据
- `start_recording()`: 开始记录
- `stop_recording()`: 停止记录
- `save_recording()`: 保存记录
- `sync_state_with_data()`: 同步状态与数据时间戳

**状态管理**:
- 使用 `DroneState` 存储无人机状态
- 线程安全的状态更新（使用 `RLock`）
- 状态历史记录（用于时间同步）
- 状态同步机制（确保数据与状态时间戳对齐）

**记录功能**:
- 支持记录图像、点云、状态
- 异步记录（后台线程）
- 数据压缩（JPEG 压缩图像，gzip 压缩点云）
- 状态同步记录（记录时附带同步的状态快照）

#### 2.2 TopicServiceManager (`topic_service_manager.py`)

管理 ROS 主题和服务的创建和生命周期。

**主要功能**:
- 主题缓存（避免重复创建）
- 服务缓存
- 线程安全的访问
- 自动清理（关闭时取消订阅）

**使用示例**:
```python
ts_mgr = TopicServiceManager(ros_instance, "connection_id")

# 获取主题
topic = ts_mgr.topic("/camera/image", "sensor_msgs/Image")
topic.subscribe(callback)

# 获取服务
service = ts_mgr.service("/set_mode", "mavros_msgs/SetMode")
response = service.call(request)
```

#### 2.3 Recorder (`recorder.py`)

高性能数据记录器，支持异步记录和压缩。

**主要功能**:
- 异步记录（后台线程处理）
- 数据压缩（图像 JPEG，点云 gzip）
- 批量写入（提高性能）
- 状态同步记录
- 支持保存为 msgpack 或 JSON 格式

**数据格式**:
```python
RecordEntry:
    - timestamp: float
    - data_type: str  # 'image', 'pointcloud', 'state'
    - data: Any       # 压缩后的数据
    - metadata: dict   # 包含同步状态等信息
```

**使用示例**:
```python
# 在客户端中自动使用
client.start_recording(
    record_images=True,
    record_pointclouds=True,
    record_states=True,
    image_quality=85
)

# 保存记录
client.save_recording("recording.rosrec", compress=True)
```

#### 2.4 RecordPlayer (`player.py`)

数据回放器，支持时间同步回放。

**主要功能**:
- 时间同步回放
- 可调节回放速度
- 支持循环播放
- 支持暂停/恢复
- 支持跳转到指定时间
- 回调机制（图像、点云、状态）

**使用示例**:
```python
from rosclient.core import Recorder, RecordPlayer

# 加载记录
recorder = Recorder.load("recording.rosrec")

# 创建播放器
player = RecordPlayer(recorder, playback_speed=1.0, loop=True)

# 设置回调
def on_image(image, timestamp):
    print(f"Image at {timestamp}")

player.set_image_callback(on_image)

# 开始播放
player.play()
```

### 3. models/ - 数据模型层

#### 3.1 DroneState (`drone.py`)

无人机状态数据模型。

**字段**:
```python
@dataclass
class DroneState:
    connected: bool = False      # 连接状态
    armed: bool = False          # 解锁状态
    mode: str = ""               # 飞行模式
    battery: float = 100.0      # 电池电量（百分比）
    latitude: float = 0.0        # 纬度
    longitude: float = 0.0       # 经度
    altitude: float = 0.0        # 高度
    roll: float = 0.0            # 横滚角（度）
    pitch: float = 0.0           # 俯仰角（度）
    yaw: float = 0.0            # 偏航角（度）
    landed: bool = True          # 是否着陆
    reached: bool = False        # 是否到达目标
    returned: bool = False       # 是否返航
    tookoff: bool = False        # 是否起飞
    last_updated: float          # 最后更新时间戳
```

#### 3.2 RosTopic (`drone.py`)

ROS 主题信息模型。

**字段**:
```python
@dataclass
class RosTopic:
    name: str                    # 主题名称
    type: str                    # 主题类型
    last_message: Optional[Dict] # 最后一条消息
    last_message_time: float     # 最后消息时间
```

#### 3.3 Waypoint (`drone.py`)

航点数据模型，用于路径规划。

#### 3.4 ConnectionState (`state.py`)

连接状态枚举。

**状态值**:
- `CONNECTING`: 连接中
- `CONNECTED`: 已连接
- `DISCONNECTED`: 已断开
- `ERROR`: 错误
- `TIMEOUT`: 超时
- `CLOSED`: 已关闭
- `RECONNECTING`: 重连中
- `RECONNECTED`: 已重连

### 4. processors/ - 数据处理器层

#### 4.1 ImageProcessor (`image_processor.py`)

智能图像处理器，支持多种 ROS 图像消息格式。

**主要功能**:
- 自动检测消息类型（CompressedImage, Image）
- 多解码器支持（按优先级尝试）
- 后处理管道（格式转换、缩放、归一化）
- 插件系统（支持算法插件）
- 结果缓存

**解码器**:
1. `CompressedImageDecoder`: 处理压缩图像
2. `RawImageDecoder`: 处理原始图像
3. `LegacyDecoder`: 向后兼容解码器

**使用示例**:
```python
from rosclient.processors import ImageProcessor

processor = ImageProcessor()

# 处理消息
result = processor.process(msg, apply_plugins=True)
if result:
    image, timestamp, plugin_results = result
    # image: numpy array (BGR format)
    # plugin_results: dict with plugin outputs
```

**插件系统**:
```python
from rosclient.processors import ImageProcessor, YOLOPlugin, SAM3Plugin

processor = ImageProcessor()

# 注册插件
yolo_plugin = YOLOPlugin(model_path="yolo.onnx")
processor.register_plugin(yolo_plugin)

sam3_plugin = SAM3Plugin(text_prompt="person")
processor.register_plugin(sam3_plugin)

# 处理图像（会自动应用插件）
result = processor.process(msg, apply_plugins=True)
```

#### 4.2 PointCloudProcessor (`pointcloud_processor.py`)

点云处理器，处理 ROS PointCloud2 消息。

**主要功能**:
- 解码 PointCloud2 消息
- 提取 x, y, z 坐标
- 返回 numpy 数组格式

**使用示例**:
```python
from rosclient.processors import PointCloudProcessor

processor = PointCloudProcessor()
result = processor.process(msg)
if result:
    points, timestamp = result
    # points: numpy array (N, 3)
```

#### 4.3 Plugins (`plugins.py`)

算法插件实现。

**YOLOPlugin**: YOLO 目标检测插件
- 支持 ONNX 和 Darknet 模型
- 返回检测框、置信度、类别

**SAM3Plugin**: SAM3 图像分割插件
- 基于文本提示的分割
- 支持输出分割图像
- 可启用/禁用

**DummyPlugin**: 测试用插件

**自定义插件**:
```python
from rosclient.processors import AlgorithmPlugin
import numpy as np

class MyPlugin(AlgorithmPlugin):
    def is_ready(self) -> bool:
        return True
    
    def process(self, image: np.ndarray, metadata: dict = None) -> dict:
        # 处理逻辑
        return {"result": "processed"}
```

### 5. utils/ - 工具函数层

#### 5.1 Logger (`logger.py`)

日志工具，提供统一的日志接口。

#### 5.2 Backoff (`backoff.py`)

重试策略工具，实现指数退避算法。

---

## 核心类和方法

### RosClientBase

所有客户端的基类，提供通用功能。

#### 连接管理

```python
def connect_async(self) -> None:
    """异步连接（抽象方法）"""
    pass

def terminate(self) -> None:
    """终止连接（抽象方法）"""
    pass

def is_connected(self) -> bool:
    """检查连接状态"""
    return self._connection_state == ConnectionState.CONNECTED
```

#### 状态访问

```python
def get_status(self) -> DroneState:
    """获取当前状态"""
    return self._state

def get_position(self) -> tuple[float, float, float]:
    """获取位置 (lat, lon, alt)"""
    return (self._state.latitude, self._state.longitude, self._state.altitude)

def get_orientation(self) -> tuple[float, float, float]:
    """获取姿态 (roll, pitch, yaw)"""
    return (self._state.roll, self._state.pitch, self._state.yaw)
```

#### 数据更新

```python
def update_odom(self, msg: Dict[str, Any]) -> None:
    """更新里程计数据"""
    # 从 ROS 消息中提取位置和姿态
    # 更新内部状态
    # 添加到状态历史
```

#### 记录功能

```python
def start_recording(
    self,
    record_images: bool = True,
    record_pointclouds: bool = True,
    record_states: bool = True,
    image_quality: int = 85,
    **kwargs
) -> bool:
    """开始记录"""
    pass

def stop_recording(self) -> bool:
    """停止记录"""
    pass

def save_recording(self, file_path: str, compress: bool = True) -> bool:
    """保存记录"""
    pass

def is_recording(self) -> bool:
    """检查是否正在记录"""
    pass
```

#### 状态同步

```python
def sync_state_with_data(self, data_timestamp: float) -> DroneState:
    """获取与数据时间戳同步的状态"""
    # 从状态历史中找到最接近的状态
    # 返回状态快照
    pass
```

### RosClient

真实 ROS 客户端实现。

#### 连接

```python
client = RosClient("ws://localhost:9090", config={
    "connect_max_retries": 5,
    "connect_backoff_seconds": 1.0
})
client.connect_async()
```

#### 数据获取

```python
# 获取最新图像（非阻塞）
image, timestamp = client.get_latest_image()

# 同步获取图像（阻塞）
image, timestamp = client.fetch_camera_image()

# 获取最新点云
points, timestamp = client.get_latest_point_cloud()

# 同步获取点云
points, timestamp = client.fetch_point_cloud()
```

#### 服务调用

```python
response = client.service_call(
    service_name="/mavros/cmd/arming",
    service_type="mavros_msgs/CommandBool",
    payload={"value": True},
    timeout=5.0
)
```

#### 发布消息

```python
client.publish(
    topic_name="/mavros/setpoint_position/local",
    topic_type="geometry_msgs/PoseStamped",
    message={
        "pose": {
            "position": {"x": 1.0, "y": 2.0, "z": 3.0}
        }
    }
)
```

---

## 使用示例

### 基本使用

```python
from rosclient import RosClient
import time

# 创建客户端
client = RosClient("ws://localhost:9090")

# 连接
client.connect_async()

# 等待连接
while not client.is_connected():
    time.sleep(0.1)

# 获取状态
state = client.get_status()
print(f"Battery: {state.battery}%")
print(f"Position: {state.latitude}, {state.longitude}")

# 获取图像
image, timestamp = client.get_latest_image()
if image is not None:
    print(f"Image shape: {image.shape}")

# 清理
client.terminate()
```

### 数据记录

```python
from rosclient import RosClient

client = RosClient("ws://localhost:9090")
client.connect_async()

# 等待连接
import time
while not client.is_connected():
    time.sleep(0.1)

# 开始记录
client.start_recording(
    record_images=True,
    record_pointclouds=True,
    record_states=True,
    image_quality=85
)

# 运行一段时间...
time.sleep(60)

# 停止并保存
client.stop_recording()
client.save_recording("flight_recording.rosrec", compress=True)

client.terminate()
```

### 数据回放

```python
from rosclient.core import Recorder, RecordPlayer
import cv2

# 加载记录
recorder = Recorder.load("flight_recording.rosrec")

# 创建播放器
player = RecordPlayer(recorder, playback_speed=1.0, loop=False)

# 设置图像回调
def on_image(image, timestamp):
    # 显示图像
    cv2.imshow("Playback", image)
    cv2.waitKey(1)

player.set_image_callback(on_image)

# 开始播放
player.play()

# 等待播放完成
while player.is_playing():
    time.sleep(0.1)

player.stop()
```

### 使用模拟客户端

```python
from rosclient import MockRosClient

# 使用模拟数据
client = MockRosClient("mock://test")

# 或使用真实数据文件
config = {
    "real_image_path": "path/to/images/",
    "real_pointcloud_path": "path/to/pointclouds/",
    "image_update_interval": 0.033  # 30 FPS
}
client = MockRosClient("mock://test", config=config)

# 或回放记录文件
config = {
    "playback_file": "recording.rosrec",
    "playback_speed": 1.0,
    "playback_loop": True
}
client = MockRosClient("mock://test", config=config)

# 使用方式与真实客户端相同
image, timestamp = client.get_latest_image()
```

### 使用图像处理器插件

```python
from rosclient import RosClient
from rosclient.processors import SAM3Plugin

client = RosClient("ws://localhost:9090")
client.connect_async()

# 获取图像处理器
image_processor = client._image_processor

# 注册 SAM3 插件
sam3_plugin = SAM3Plugin(
    text_prompt="person",
    enabled=True,
    output_segmented_image=True
)
image_processor.register_plugin(sam3_plugin)

# 处理图像（会自动应用插件）
image, timestamp = client.get_latest_image()
if image is not None:
    # 图像已经过 SAM3 处理
    # 可以通过 processor 获取分割结果
    segmented = image_processor.get_segmented_image()
```

---

## 扩展指南

### 创建自定义客户端

```python
from rosclient.core import RosClientBase
from rosclient.models.state import ConnectionState

class MyCustomClient(RosClientBase):
    def __init__(self, connection_str: str, config: dict = None):
        super().__init__(connection_str, config)
        # 初始化自定义资源
    
    def connect_async(self) -> None:
        """实现连接逻辑"""
        # 异步连接
        def connect_task():
            # 连接代码
            with self._lock:
                self._connection_state = ConnectionState.CONNECTED
        
        import threading
        thread = threading.Thread(target=connect_task, daemon=True)
        thread.start()
    
    def terminate(self) -> None:
        """实现断开逻辑"""
        with self._lock:
            self._connection_state = ConnectionState.DISCONNECTED
            # 清理资源
```

### 创建自定义图像插件

```python
from rosclient.processors import AlgorithmPlugin
import numpy as np
import cv2

class MyImagePlugin(AlgorithmPlugin):
    def __init__(self, param1: str, param2: int):
        self.param1 = param1
        self.param2 = param2
        self._initialized = False
    
    def is_ready(self) -> bool:
        """检查插件是否就绪"""
        if not self._initialized:
            # 初始化逻辑
            self._initialized = True
        return self._initialized
    
    def process(self, image: np.ndarray, metadata: dict = None) -> dict:
        """处理图像"""
        # 处理逻辑
        result = {
            "status": "success",
            "data": processed_data
        }
        return result
```

### 创建自定义数据处理器

```python
from rosclient.processors import ImageProcessor

class MyImageProcessor(ImageProcessor):
    def __init__(self, logger=None, config=None):
        super().__init__(logger, config)
        # 添加自定义解码器
        self._decoders.insert(0, MyCustomDecoder())
    
    def process(self, msg: dict, apply_plugins: bool = True, metadata: dict = None):
        # 自定义处理逻辑
        result = super().process(msg, apply_plugins, metadata)
        # 后处理
        return result
```

### 扩展记录格式

```python
from rosclient.core import Recorder, RecordEntry

class MyRecorder(Recorder):
    def record_custom_data(self, data: Any, timestamp: float) -> bool:
        """记录自定义数据"""
        entry = RecordEntry(
            timestamp=timestamp,
            data_type="custom",
            data=data,
            metadata={}
        )
        try:
            self._record_queue.put_nowait(entry)
            return True
        except queue.Full:
            return False
```

---

## API 参考

### RosClientBase

#### 方法

- `connect_async() -> None`: 异步连接
- `terminate() -> None`: 终止连接
- `is_connected() -> bool`: 检查连接状态
- `get_status() -> DroneState`: 获取状态
- `get_position() -> tuple[float, float, float]`: 获取位置
- `get_orientation() -> tuple[float, float, float]`: 获取姿态
- `update_odom(msg: Dict) -> None`: 更新里程计
- `start_recording(...) -> bool`: 开始记录
- `stop_recording() -> bool`: 停止记录
- `save_recording(file_path: str, compress: bool) -> bool`: 保存记录
- `is_recording() -> bool`: 检查是否记录中
- `sync_state_with_data(timestamp: float) -> DroneState`: 同步状态

### RosClient

#### 方法

- `get_latest_image() -> Optional[Tuple[np.ndarray, float]]`: 获取最新图像
- `fetch_camera_image() -> Optional[Tuple[np.ndarray, float]]`: 同步获取图像
- `get_latest_point_cloud() -> Optional[Tuple[np.ndarray, float]]`: 获取最新点云
- `fetch_point_cloud() -> Optional[Tuple[np.ndarray, float]]`: 同步获取点云
- `service_call(service_name: str, service_type: str, payload: Dict, ...) -> Dict`: 调用服务
- `publish(topic_name: str, topic_type: str, message: Dict, ...) -> None`: 发布消息

### Recorder

#### 方法

- `start_recording(...) -> None`: 开始记录
- `stop_recording() -> None`: 停止记录
- `is_recording() -> bool`: 检查是否记录中
- `record_image(image: np.ndarray, timestamp: float, state: DroneState) -> bool`: 记录图像
- `record_pointcloud(points: np.ndarray, timestamp: float, state: DroneState) -> bool`: 记录点云
- `record_state(state: DroneState, timestamp: float) -> bool`: 记录状态
- `save(file_path: str, compress: bool) -> bool`: 保存记录
- `load(file_path: str) -> Recorder`: 加载记录（类方法）
- `get_statistics() -> Dict`: 获取统计信息

### RecordPlayer

#### 方法

- `play(start_time: float = None) -> None`: 开始播放
- `pause() -> None`: 暂停
- `resume() -> None`: 恢复
- `stop() -> None`: 停止
- `seek(timestamp: float) -> bool`: 跳转到指定时间
- `is_playing() -> bool`: 检查是否播放中
- `is_paused() -> bool`: 检查是否暂停
- `get_progress() -> float`: 获取播放进度
- `get_current_time() -> float`: 获取当前时间
- `set_image_callback(callback: Callable) -> None`: 设置图像回调
- `set_pointcloud_callback(callback: Callable) -> None`: 设置点云回调
- `set_state_callback(callback: Callable) -> None`: 设置状态回调

### ImageProcessor

#### 方法

- `process(msg: Dict, apply_plugins: bool, metadata: Dict) -> Optional[Tuple]`: 处理消息
- `process_simple(msg: Dict) -> Optional[Tuple]`: 简单处理（无插件）
- `register_plugin(plugin: AlgorithmPlugin) -> None`: 注册插件
- `unregister_plugin(plugin: AlgorithmPlugin) -> None`: 注销插件
- `decode_message(msg: Dict) -> Optional[np.ndarray]`: 解码消息

---

## 最佳实践

### 1. 连接管理

```python
# 使用上下文管理器模式
class RosClientContext:
    def __init__(self, connection_str: str):
        self.client = RosClient(connection_str)
    
    def __enter__(self):
        self.client.connect_async()
        # 等待连接
        import time
        timeout = 10
        start = time.time()
        while not self.client.is_connected() and (time.time() - start) < timeout:
            time.sleep(0.1)
        if not self.client.is_connected():
            raise ConnectionError("Failed to connect")
        return self.client
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.terminate()

# 使用
with RosClientContext("ws://localhost:9090") as client:
    image, timestamp = client.get_latest_image()
```

### 2. 错误处理

```python
try:
    client = RosClient("ws://localhost:9090")
    client.connect_async()
    
    # 操作
    image, timestamp = client.get_latest_image()
    
except ConnectionError as e:
    print(f"Connection error: {e}")
except Exception as e:
    print(f"Error: {e}")
finally:
    client.terminate()
```

### 3. 数据记录

```python
# 在连接后立即开始记录
client.connect_async()
while not client.is_connected():
    time.sleep(0.1)

# 开始记录
client.start_recording(
    record_images=True,
    record_pointclouds=True,
    record_states=True
)

# 定期检查记录状态
if client.is_recording():
    stats = client.get_recording_statistics()
    print(f"Recorded: {stats['images_recorded']} images")
```

### 4. 性能优化

```python
# 使用缓存获取最新数据（非阻塞）
image, timestamp = client.get_latest_image()

# 避免频繁同步获取（阻塞）
# image, timestamp = client.fetch_camera_image()  # 仅在必要时使用

# 调整记录参数
client.start_recording(
    image_quality=75,  # 降低质量以提高性能
    max_queue_size=50  # 减小队列大小
)
```

---

## 故障排除

### 连接问题

1. **无法连接**: 检查 rosbridge 是否运行，端口是否正确
2. **连接超时**: 增加 `connect_max_retries` 和 `connect_backoff_max`
3. **频繁断开**: 检查网络稳定性，考虑实现自动重连

### 数据问题

1. **图像解码失败**: 检查消息格式，确认解码器支持该格式
2. **点云数据为空**: 检查 PointCloud2 消息格式，确认字段正确
3. **状态不同步**: 启用状态同步功能，检查时间戳

### 性能问题

1. **内存占用高**: 减小缓存队列大小，降低图像质量
2. **CPU 占用高**: 减少插件使用，优化处理逻辑
3. **记录文件过大**: 使用压缩，降低图像质量

---

## 版本历史

- **v1.0.0**: 初始版本
  - 基本 ROS 客户端功能
  - 数据记录和回放
  - 图像和点云处理
  - 插件系统

---

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

---

## 许可证

[根据项目实际情况填写]

---

## 联系方式

[根据项目实际情况填写]

