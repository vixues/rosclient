# SAM3 Plugin 使用指南

## 概述

SAM3Plugin 集成了 Segment Anything Model 3，支持基于文本提示的图像分割功能。

## 安装依赖

```bash
pip install sam3
# 或
pip install torch torchvision  # SAM3 需要 PyTorch
pip install sam3
```

## 基本使用

### 1. 初始化并注册插件

```python
from rosclient.processors import ImageProcessor, SAM3Plugin
from rosclient.utils.logger import setup_logger

# 初始化处理器
logger = setup_logger("ImageProcessor")
processor = ImageProcessor(logger=logger)

# 创建并注册 SAM3 插件
sam3 = SAM3Plugin(
    text_prompt="person",  # 文本提示，例如 "person", "car", "dog"
    enabled=True,          # 启用处理
    output_segmented_image=True,  # 输出分割后的图像
    mask_threshold=0.5,    # 掩码阈值
    logger=logger
)
processor.register_plugin(sam3)
```

### 2. 处理图像

```python
# 处理 ROS 消息
result = processor.process(ros_message, apply_plugins=True)
if result:
    segmented_image, timestamp, plugin_results = result
    
    # segmented_image 是分割后的图像（如果启用）
    # 原始图像会被分割结果覆盖
    
    # 获取 SAM3 的详细结果
    sam3_result = plugin_results.get("SAM3Plugin", {})
    if sam3_result.get("enabled", False):
        masks = sam3_result.get("masks", [])
        boxes = sam3_result.get("boxes", [])
        scores = sam3_result.get("scores", [])
        num_segments = sam3_result.get("num_segments", 0)
        
        print(f"Found {num_segments} segments")
```

### 3. 动态控制处理

```python
# 禁用 SAM3 处理
sam3.disable()
# 或
processor.set_plugin_enabled("SAM3Plugin", False)

# 启用 SAM3 处理
sam3.enable()
# 或
processor.set_plugin_enabled("SAM3Plugin", True)

# 检查是否启用
is_enabled = processor.get_plugin_enabled("SAM3Plugin")
```

### 4. 更新文本提示

```python
# 更改分割提示
sam3.set_text_prompt("car")  # 从 "person" 改为 "car"

# 处理新图像时会使用新的提示
result = processor.process(new_message, apply_plugins=True)
```

### 5. 获取分割后的图像

```python
# 方法1: 从处理结果中获取（已自动应用）
result = processor.process(msg, apply_plugins=True)
if result:
    segmented_image, timestamp, results = result
    # segmented_image 已经是分割后的图像

# 方法2: 从缓存中获取
segmented_image = processor.get_segmented_image()
if segmented_image is not None:
    # 使用分割后的图像
    cv2.imshow("Segmented", segmented_image)
```

## 配置选项

### SAM3Plugin 参数

- `text_prompt` (str): 文本提示，描述要分割的对象
- `enabled` (bool): 是否启用处理（默认: True）
- `output_segmented_image` (bool): 是否在结果中包含分割后的图像（默认: True）
- `mask_threshold` (float): 掩码二值化阈值，范围 0-1（默认: 0.5）
- `logger` (Logger): 可选的日志记录器

### 输出格式

SAM3Plugin 返回的字典包含：

```python
{
    "enabled": True,                    # 是否启用
    "masks": [...],                     # 分割掩码列表
    "boxes": [[x1, y1, x2, y2], ...],  # 边界框列表
    "scores": [0.95, 0.87, ...],       # 置信度分数列表
    "num_segments": 3,                  # 分割数量
    "text_prompt": "person",            # 使用的文本提示
    "segmented_image": np.ndarray       # 分割后的图像（如果启用）
}
```

## 性能优化

1. **按需启用**: 只在需要时启用 SAM3 处理
   ```python
   # 订阅时禁用（快速）
   processor.set_plugin_enabled("SAM3Plugin", False)
   
   # 需要分割时启用
   processor.set_plugin_enabled("SAM3Plugin", True)
   result = processor.process(msg, apply_plugins=True)
   ```

2. **缓存利用**: 使用缓存避免重复处理
   ```python
   config = {"enable_cache": True}
   processor = ImageProcessor(config=config)
   ```

3. **批量处理**: 对于视频流，考虑异步处理

## 完整示例

```python
from rosclient.processors import ImageProcessor, SAM3Plugin
from rosclient.clients.ros_client import RosClient
import cv2

# 初始化 ROS 客户端
client = RosClient("ws://localhost:9090")
client.connect_async()

# 初始化图像处理器
processor = ImageProcessor()

# 注册 SAM3 插件
sam3 = SAM3Plugin(
    text_prompt="person",
    enabled=True,
    output_segmented_image=True
)
processor.register_plugin(sam3)

# 处理图像
def on_image_received(msg):
    result = processor.process(msg, apply_plugins=True)
    if result:
        segmented_image, timestamp, results = result
        
        # 显示分割后的图像
        cv2.imshow("Segmented Image", segmented_image)
        cv2.waitKey(1)
        
        # 获取详细信息
        sam3_result = results.get("SAM3Plugin", {})
        if sam3_result.get("enabled", False):
            print(f"Found {sam3_result['num_segments']} segments")

# 订阅相机话题
# client.subscribe_camera(on_image_received)
```

## 注意事项

1. **模型加载**: SAM3 模型在首次使用时加载，可能需要一些时间
2. **内存使用**: SAM3 模型较大，注意内存使用
3. **处理速度**: 分割处理相对较慢，建议按需启用
4. **文本提示**: 使用清晰、具体的文本提示可以获得更好的分割效果

## 故障排除

### 模型加载失败

```python
# 检查模型是否就绪
if not sam3.is_ready():
    print("SAM3 model not ready, check installation")
```

### 没有分割结果

```python
# 检查是否启用
if not sam3.enabled:
    sam3.enable()

# 检查文本提示
if not sam3.text_prompt:
    sam3.set_text_prompt("your_prompt")
```

### 性能问题

```python
# 禁用输出分割图像以提升性能
sam3 = SAM3Plugin(
    text_prompt="person",
    output_segmented_image=False  # 只返回结果，不生成图像
)
```

