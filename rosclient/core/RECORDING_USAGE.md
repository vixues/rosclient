# 记录模块使用指南

## 概述

记录模块提供了高性能的数据录制和播放功能，支持：
- **图像录制**：自动压缩为 JPEG 格式
- **点云录制**：使用 gzip 压缩
- **状态录制**：记录无人机状态信息
- **异步录制**：后台线程处理，不影响主程序性能
- **时间同步播放**：支持按原始时间戳播放

## 基本使用

### 1. 开始录制

```python
from rosclient.clients import RosClient, MockRosClient

# 创建客户端
client = RosClient("ws://localhost:9090")
# 或使用 MockRosClient
# client = MockRosClient("mock://test")

# 连接到 ROS
client.connect_async()

# 等待连接
import time
time.sleep(2)

# 开始录制
client.start_recording(
    record_images=True,      # 录制图像
    record_pointclouds=True, # 录制点云
    record_states=True,      # 录制状态
    image_quality=85         # JPEG 质量 (1-100)
)

# 正常使用客户端，数据会自动记录
for i in range(100):
    image = client.get_latest_image()
    pointcloud = client.get_latest_point_cloud()
    state = client.get_status()
    time.sleep(0.1)

# 停止录制
client.stop_recording()

# 保存录制
client.save_recording("recording.rosrec", compress=True)
```

### 2. 加载和播放录制

```python
from rosclient.core import Recorder, RecordPlayer
import numpy as np

# 加载录制
recorder = Recorder.load("recording.rosrec")

if recorder:
    # 创建播放器
    player = RecordPlayer(
        recorder,
        playback_speed=1.0,  # 1.0 = 实时播放，2.0 = 2倍速
        loop=False           # 是否循环播放
    )
    
    # 设置回调函数
    def on_image(image: np.ndarray, timestamp: float):
        print(f"Image received at {timestamp}")
        # 处理图像...
    
    def on_pointcloud(points: np.ndarray, timestamp: float):
        print(f"Point cloud received at {timestamp}, {len(points)} points")
        # 处理点云...
    
    def on_state(state, timestamp: float):
        print(f"State updated: mode={state.mode}, battery={state.battery}")
        # 处理状态...
    
    player.set_image_callback(on_image)
    player.set_pointcloud_callback(on_pointcloud)
    player.set_state_callback(on_state)
    
    # 开始播放
    player.play()
    
    # 等待播放完成
    import time
    while player.is_playing():
        time.sleep(0.1)
        print(f"Progress: {player.get_progress()*100:.1f}%")
    
    # 停止播放
    player.stop()
```

### 3. 手动访问录制数据

```python
from rosclient.core import Recorder

# 加载录制
recorder = Recorder.load("recording.rosrec")

if recorder:
    # 获取所有图像
    images = recorder.get_entries(data_type="image")
    print(f"Total images: {len(images)}")
    
    # 解码单个条目
    for entry in images[:5]:  # 前5个图像
        result = recorder.decode_entry(entry)
        if result:
            image, timestamp = result
            print(f"Image shape: {image.shape}, timestamp: {timestamp}")
    
    # 获取特定时间范围的数据
    start_time = recorder._metadata.start_time
    end_time = start_time + 10.0  # 前10秒
    
    entries = recorder.get_entries(
        start_time=start_time,
        end_time=end_time
    )
    print(f"Entries in first 10 seconds: {len(entries)}")
```

### 4. 使用播放器手动播放

```python
from rosclient.core import Recorder, RecordPlayer

recorder = Recorder.load("recording.rosrec")
player = RecordPlayer(recorder)

# 手动获取特定索引的条目
for i in range(10):
    result = player.get_entry_at_index(i)
    if result:
        data, timestamp = result
        print(f"Entry {i}: {type(data)}, timestamp: {timestamp}")

# 获取所有图像
all_images = player.get_all_images()
print(f"Total images: {len(all_images)}")

# 获取所有点云
all_pointclouds = player.get_all_pointclouds()
print(f"Total point clouds: {len(all_pointclouds)}")

# 获取所有状态
all_states = player.get_all_states()
print(f"Total states: {len(all_states)}")
```

## 高级功能

### 1. 配置录制参数

```python
# 在创建客户端时配置
config = {
    "recording": {
        "max_queue_size": 200,  # 队列大小
        "batch_size": 20         # 批处理大小
    }
}

client = RosClient("ws://localhost:9090", config=config)
```

### 2. 获取录制统计信息

```python
# 录制过程中
stats = client.get_recording_statistics()
print(f"Images recorded: {stats['images_recorded']}")
print(f"Point clouds recorded: {stats['pointclouds_recorded']}")
print(f"States recorded: {stats['states_recorded']}")
print(f"Dropped: {stats['dropped']}")

# 播放过程中
player_stats = player.get_statistics()
print(f"Progress: {player_stats['progress']*100:.1f}%")
print(f"Playback speed: {player_stats['playback_speed']}")
```

### 3. 控制播放

```python
# 暂停和恢复
player.pause()
time.sleep(2)
player.resume()

# 跳转到特定时间
player.stop()
player.seek(30.0)  # 跳转到30秒
player.play()

# 改变播放速度
player.playback_speed = 2.0  # 2倍速
```

## 性能优化建议

1. **图像质量**：根据需求调整 `image_quality`（默认85），较低的质量可以减小文件大小
2. **批处理大小**：增加 `batch_size` 可以提高写入性能，但会增加内存使用
3. **队列大小**：增加 `max_queue_size` 可以减少数据丢失，但会增加内存使用
4. **压缩**：保存时使用 `compress=True` 可以显著减小文件大小

## 文件格式

录制文件使用以下格式：
- **msgpack + gzip**（如果 msgpack 可用）：二进制格式，高效压缩
- **JSON + gzip**（回退方案）：文本格式，兼容性更好

文件包含：
- 元数据：录制开始/结束时间、客户端类型、配置等
- 条目列表：每个条目包含时间戳、数据类型、压缩数据和元数据

## 注意事项

1. 录制会消耗内存，长时间录制建议定期保存
2. 播放时回调函数应该快速执行，避免阻塞播放线程
3. 如果队列满了，新数据会被丢弃（记录在统计信息中）
4. 确保有足够的磁盘空间保存录制文件

