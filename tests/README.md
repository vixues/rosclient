# 测试说明

## 测试结构

```text
tests/
├── conftest.py              # pytest配置和共享fixtures
├── tests_models/            # 模型测试
│   ├── test_drone.py        # DroneState和RosTopic测试
│   └── test_state.py        # ConnectionState测试
├── tests_utils/             # 工具函数测试
│   ├── test_backoff.py      # 指数退避算法测试
│   └── test_logger.py       # 日志工具测试
├── tests_core/              # 核心功能测试
│   ├── test_base.py         # RosClientBase测试
│   └── test_topic_service_manager.py  # TopicServiceManager测试
└── tests_clients/           # 客户端测试
    ├── test_mock_client.py  # MockRosClient测试
    └── test_ros_client.py  # RosClient测试
```

## 运行测试

### 运行所有测试

```bash
pytest
```

### 运行特定模块的测试

```bash
pytest tests/tests_models/
pytest tests/tests_utils/
pytest tests/tests_core/
pytest tests/tests_clients/
```

### 运行特定测试文件

```bash
pytest tests/tests_models/test_drone.py
```

### 运行特定测试函数

```bash
pytest tests/tests_models/test_drone.py::TestDroneState::test_default_initialization
```

### 带覆盖率报告

```bash
pytest --cov=rosclient --cov-report=html
```

### 详细输出

```bash
pytest -v
```

### 显示打印输出

```bash
pytest -s
```

## 测试标记

- `@pytest.mark.unit` - 单元测试
- `@pytest.mark.integration` - 集成测试
- `@pytest.mark.slow` - 慢速测试

使用标记运行：

```bash
pytest -m unit
pytest -m "not slow"
```

## 测试覆盖率目标

目标覆盖率：>80%

查看覆盖率报告：

```bash
pytest --cov=rosclient --cov-report=html
# 然后打开 htmlcov/index.html
```

