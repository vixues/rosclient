"""Pytest configuration and shared fixtures."""
import pytest
from typing import Dict, Any

from rosclient.clients.mock_client import MockRosClient
from rosclient.models.drone import DroneState, RosTopic
from rosclient.models.state import ConnectionState


@pytest.fixture
def mock_client():
    """Create a mock ROS client for testing."""
    return MockRosClient("ws://localhost:9090")


@pytest.fixture
def mock_client_with_config():
    """Create a mock ROS client with custom config."""
    config = {
        "service_call_timeout": 10.0,
        "connect_max_retries": 3,
    }
    return MockRosClient("ws://localhost:9090", config=config)


@pytest.fixture
def sample_drone_state():
    """Create a sample drone state."""
    return DroneState(
        connected=True,
        armed=True,
        mode="GUIDED",
        battery=85.5,
        latitude=22.5329,
        longitude=113.93029,
        altitude=100.0,
        roll=1.5,
        pitch=2.3,
        yaw=45.0,
        landed=False,
        reached=True,
        returned=False,
        tookoff=True,
    )


@pytest.fixture
def sample_ros_topic():
    """Create a sample ROS topic."""
    return RosTopic(
        name="/mavros/state",
        type="mavros_msgs/State",
        last_message={"armed": True, "mode": "GUIDED"},
    )


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample configuration dictionary."""
    return {
        "connect_max_retries": 5,
        "connect_backoff_seconds": 1.0,
        "connect_backoff_max": 30.0,
        "service_call_timeout": 5.0,
        "service_call_retries": 2,
        "publish_retries": 2,
        "logger_level": 20,
    }

