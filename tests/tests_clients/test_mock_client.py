"""Tests for MockRosClient."""
import pytest
import time

from rosclient.clients.mock_client import MockRosClient
from rosclient.models.state import ConnectionState


class TestMockRosClient:
    """Tests for MockRosClient."""

    def test_initialization(self):
        """Test MockRosClient initialization."""
        client = MockRosClient("ws://localhost:9090")
        assert client.connection_str == "ws://localhost:9090"
        assert client.is_connected() is True
        assert client._state.connected is True
        assert client._state.battery == 100.0

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = {"service_call_timeout": 10.0}
        client = MockRosClient("ws://localhost:9090", config=config)
        assert client._config["service_call_timeout"] == 10.0

    def test_connect_async(self):
        """Test connect_async method."""
        client = MockRosClient("ws://localhost:9090")
        client.terminate()
        assert client.is_connected() is False
        
        client.connect_async()
        assert client.is_connected() is True
        assert client._connection_state == ConnectionState.CONNECTED

    def test_terminate(self):
        """Test terminate method."""
        client = MockRosClient("ws://localhost:9090")
        assert client.is_connected() is True
        
        client.terminate()
        assert client.is_connected() is False
        assert client._connection_state == ConnectionState.DISCONNECTED

    def test_service_call(self):
        """Test service_call method."""
        client = MockRosClient("ws://localhost:9090")
        
        response = client.service_call("/test/service", "std_srvs/Empty", {"data": "test"})
        
        assert response["mock"] == "success"
        assert response["service"] == "/test/service"
        assert len(client.service_calls) == 1
        assert client.service_calls[0]["service_name"] == "/test/service"

    def test_publish(self):
        """Test publish method."""
        client = MockRosClient("ws://localhost:9090")
        
        message = {"data": "test"}
        client.publish("/test/topic", "std_msgs/String", message)
        
        assert len(client.published_messages) == 1
        assert client.published_messages[0]["topic_name"] == "/test/topic"
        assert client.published_messages[0]["message"] == message

    def test_set_mode(self):
        """Test set_mode helper method."""
        client = MockRosClient("ws://localhost:9090")
        
        client.set_mode("GUIDED")
        assert client.get_status().mode == "GUIDED"

    def test_set_armed(self):
        """Test set_armed helper method."""
        client = MockRosClient("ws://localhost:9090")
        
        client.set_armed(True)
        assert client.get_status().armed is True
        
        client.set_armed(False)
        assert client.get_status().armed is False

    def test_set_battery(self):
        """Test set_battery helper method."""
        client = MockRosClient("ws://localhost:9090")
        
        client.set_battery(75.5)
        assert client.get_status().battery == 75.5

    def test_set_position(self):
        """Test set_position helper method."""
        client = MockRosClient("ws://localhost:9090")
        
        client.set_position(22.5329, 113.93029, 100.0)
        pos = client.get_position()
        assert pos[0] == 22.5329
        assert pos[1] == 113.93029
        assert pos[2] == 100.0

    def test_get_status(self):
        """Test get_status method."""
        client = MockRosClient("ws://localhost:9090")
        state = client.get_status()
        
        assert state.connected is True
        assert state.battery == 100.0
        assert isinstance(state.last_updated, float)

    def test_get_position(self):
        """Test get_position method."""
        client = MockRosClient("ws://localhost:9090")
        pos = client.get_position()
        
        assert len(pos) == 3
        assert all(isinstance(x, float) for x in pos)

    def test_get_orientation(self):
        """Test get_orientation method."""
        client = MockRosClient("ws://localhost:9090")
        ori = client.get_orientation()
        
        assert len(ori) == 3
        assert all(isinstance(x, float) for x in ori)

