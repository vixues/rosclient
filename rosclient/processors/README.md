# 图像处理模块使用指南

## 概述

智能图像处理模块支持多种ROS图像消息格式，提供高性能解码、后处理和算法插件扩展能力。

## 功能特性

- ✅ **多格式支持**: 自动检测并处理 `sensor_msgs/Image` 和 `sensor_msgs/CompressedImage`
- ✅ **高性能解码**: 优化的解码器链，支持多种编码格式
- ✅ **后处理管道**: 尺寸调整、格式转换、归一化等
- ✅ **插件系统**: 易于集成YOLO等外部算法
- ✅ **输出控制**: 灵活的配置选项
- ✅ **性能优化**: 结果缓存、异步处理支持

## 基本使用

```python
from rosclient.processors import ImageProcessor, ImageFormat
from rosclient.utils.logger import setup_logger

# 初始化处理器
logger = setup_logger("ImageProcessor")
processor = ImageProcessor(logger=logger)

# 处理消息（简单模式）
result = processor.process_simple(ros_message)
if result:
    frame, timestamp = result
    # frame 是 BGR 格式的 numpy 数组
```

## 高级配置

```python
# 配置后处理选项
config = {
    "output_format": ImageFormat.RGB,  # 输出RGB格式
    "resize": (640, 480),              # 调整尺寸
    "keep_aspect": True,                # 保持宽高比
    "normalize": False,                 # 不归一化
    "enable_cache": True                # 启用缓存
}

processor = ImageProcessor(logger=logger, config=config)

# 完整处理（包含插件）
result = processor.process(ros_message, apply_plugins=True)
if result:
    frame, timestamp, plugin_results = result
    # plugin_results 包含所有插件的输出
```

## 集成YOLO算法

```python
from rosclient.processors import ImageProcessor, YOLOPlugin

# 初始化处理器
processor = ImageProcessor(logger=logger)

# 注册YOLO插件
yolo = YOLOPlugin(
    model_path="path/to/yolo.onnx",
    confidence_threshold=0.5,
    nms_threshold=0.4,
    logger=logger
)
processor.register_plugin(yolo)

# 处理图像（自动运行YOLO检测）
result = processor.process(ros_message, apply_plugins=True)
if result:
    frame, timestamp, results = result
    yolo_results = results.get("YOLOPlugin", {})
    
    boxes = yolo_results.get("boxes", [])
    scores = yolo_results.get("scores", [])
    classes = yolo_results.get("classes", [])
    
    # 使用检测结果...
```

## 集成SAM3分割算法

```python
from rosclient.processors import ImageProcessor, SAM3Plugin

# 初始化处理器
processor = ImageProcessor(logger=logger)

# 注册SAM3插件
sam3 = SAM3Plugin(
    text_prompt="person",  # 文本提示
    enabled=True,          # 启用处理
    output_segmented_image=True,  # 输出分割后的图像
    mask_threshold=0.5,
    logger=logger
)
processor.register_plugin(sam3)

# 处理图像（自动运行SAM3分割）
result = processor.process(ros_message, apply_plugins=True)
if result:
    segmented_image, timestamp, results = result
    # segmented_image 是分割后的图像（掩码已应用）
    
    sam3_results = results.get("SAM3Plugin", {})
    if sam3_results.get("enabled", False):
        masks = sam3_results.get("masks", [])
        boxes = sam3_results.get("boxes", [])
        scores = sam3_results.get("scores", [])
        num_segments = sam3_results.get("num_segments", 0)
        
        print(f"Found {num_segments} segments")

# 动态控制处理
sam3.disable()  # 禁用
sam3.enable()   # 启用
sam3.set_text_prompt("car")  # 更新提示

# 或通过处理器控制
processor.set_plugin_enabled("SAM3Plugin", False)
processor.set_plugin_enabled("SAM3Plugin", True)
```

详细使用说明请参考 [SAM3_USAGE.md](SAM3_USAGE.md)

## 自定义算法插件

```python
from rosclient.processors import AlgorithmPlugin
import numpy as np

class MyCustomPlugin(AlgorithmPlugin):
    def is_ready(self) -> bool:
        # 检查模型/资源是否就绪
        return True
    
    def process(self, image: np.ndarray, metadata: dict = None) -> dict:
        # 实现你的算法逻辑
        # image: BGR格式的numpy数组
        # metadata: 包含timestamp, shape等信息
        
        # 处理图像...
        result = {"custom_output": "value"}
        return result

# 注册自定义插件
processor.register_plugin(MyCustomPlugin())
```

## 消息类型检测

```python
from rosclient.processors import MessageType

msg_type = processor.detect_message_type(ros_message)
if msg_type == MessageType.COMPRESSED_IMAGE:
    print("Compressed image detected")
elif msg_type == MessageType.RAW_IMAGE:
    print("Raw image detected")
```

## 性能优化建议

1. **缓存**: 启用缓存可避免重复处理
   ```python
   config = {"enable_cache": True}
   ```

2. **选择性插件**: 只在需要时启用插件
   ```python
   # 订阅时不用插件（快速）
   result = processor.process_simple(msg)
   
   # 需要检测时再用插件
   result = processor.process(msg, apply_plugins=True)
   ```

3. **批量处理**: 对于高频图像流，考虑异步处理插件

## 支持的编码格式

### Raw Image (sensor_msgs/Image)
- `bgr8`, `rgb8`
- `bgra8`, `rgba8`
- `mono8`, `8UC1`, `8UC3`, `8UC4`
- `32FC1`, `32FC3`, `32FC4`

### Compressed Image (sensor_msgs/CompressedImage)
- JPEG
- PNG
- 其他OpenCV支持的格式

## API参考

### ImageProcessor

- `process(msg, apply_plugins=True, metadata=None)`: 完整处理
- `process_simple(msg)`: 简单处理（无插件）
- `decode_message(msg)`: 仅解码
- `register_plugin(plugin)`: 注册插件
- `unregister_plugin(plugin)`: 取消注册
- `detect_message_type(msg)`: 检测消息类型
- `update_config(config)`: 更新配置
- `get_last_result()`: 获取缓存结果
- `get_segmented_image()`: 获取分割后的图像（如果可用）
- `get_plugin_enabled(plugin_name)`: 检查插件是否启用
- `set_plugin_enabled(plugin_name, enabled)`: 启用/禁用插件

### ImageFormat

- `BGR`: OpenCV默认格式
- `RGB`: RGB格式
- `GRAY`: 灰度图
- `HSV`: HSV色彩空间

