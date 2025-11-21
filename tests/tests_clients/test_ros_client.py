"""Tests for RosClient."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import time

from rosclient.clients.ros_client import RosClient
from rosclient.models.state import ConnectionState


class TestRosClient:
    """Tests for RosClient."""

    def test_initialization_valid_url(self):
        """Test RosClient initialization with valid URL."""
        client = RosClient("ws://localhost:9090")
        assert client.connection_str == "ws://localhost:9090"
        assert client._host == "localhost"
        assert client._port == 9090

    def test_initialization_with_port(self):
        """Test RosClient initialization with custom port."""
        client = RosClient("ws://localhost:8080")
        assert client._port == 8080

    def test_initialization_default_port(self):
        """Test RosClient initialization with default port."""
        client = RosClient("ws://localhost")
        assert client._port == 9090

    def test_initialization_invalid_url(self):
        """Test RosClient initialization with invalid URL."""
        with pytest.raises(ValueError, match="Invalid WebSocket URL"):
            RosClient("http://localhost:9090")
        
        with pytest.raises(ValueError, match="Invalid WebSocket URL"):
            RosClient("invalid")

    def test_initialization_with_config(self):
        """Test RosClient initialization with custom config."""
        config = {"connect_max_retries": 3}
        client = RosClient("ws://localhost:9090", config=config)
        assert client._config["connect_max_retries"] == 3

    def test_ensure_ts_mgr_not_connected(self):
        """Test _ensure_ts_mgr raises when not connected."""
        client = RosClient("ws://localhost:9090")
        
        with pytest.raises(RuntimeError, match="Not connected"):
            client._ensure_ts_mgr()

    def test_ensure_ts_mgr_not_initialized(self):
        """Test _ensure_ts_mgr raises when not initialized."""
        client = RosClient("ws://localhost:9090")
        with patch.object(client, 'is_connected', return_value=True):
            with pytest.raises(RuntimeError, match="not initialized"):
                client._ensure_ts_mgr()

    def test_update_state(self):
        """Test update_state method."""
        client = RosClient("ws://localhost:9090")
        
        msg = {
            "armed": True,
            "mode": "GUIDED",
        }
        
        client.update_state(msg)
        state = client.get_status()
        assert state.armed is True
        assert state.mode == "GUIDED"

    def test_update_drone_state(self):
        """Test update_drone_state method."""
        client = RosClient("ws://localhost:9090")
        
        msg = {
            "landed": False,
            "returned": False,
            "reached": True,
            "tookoff": True,
        }
        
        client.update_drone_state(msg)
        state = client.get_status()
        assert state.landed is False
        assert state.reached is True
        assert state.tookoff is True

    def test_update_battery(self):
        """Test update_battery method."""
        client = RosClient("ws://localhost:9090")
        
        msg = {"percentage": 0.75}
        client.update_battery(msg)
        assert client.get_status().battery == 75.0
        
        msg = {"percent": 0.5}
        client.update_battery(msg)
        assert client.get_status().battery == 50.0
        
        msg = {"battery": 0.25}
        client.update_battery(msg)
        assert client.get_status().battery == 25.0

    def test_update_battery_percentage(self):
        """Test update_battery with percentage value."""
        client = RosClient("ws://localhost:9090")
        
        msg = {"percentage": 85.5}  # Already percentage
        client.update_battery(msg)
        assert client.get_status().battery == 85.5

    def test_update_gps(self):
        """Test update_gps method."""
        client = RosClient("ws://localhost:9090")
        
        msg = {
            "latitude": 22.5329,
            "longitude": 113.93029,
            "altitude": 100.0,
        }
        
        client.update_gps(msg)
        pos = client.get_position()
        assert pos[0] == 22.5329
        assert pos[1] == 113.93029
        assert pos[2] == 100.0

    def test_update_gps_alternate_keys(self):
        """Test update_gps with alternate key names."""
        client = RosClient("ws://localhost:9090")
        
        msg = {
            "lat": 22.5329,
            "lon": 113.93029,
            "alt": 100.0,
        }
        
        client.update_gps(msg)
        pos = client.get_position()
        assert pos[0] == 22.5329
        assert pos[1] == 113.93029
        assert pos[2] == 100.0

    def test_terminate(self):
        """Test terminate method."""
        client = RosClient("ws://localhost:9090")
        
        # Mock the connection objects
        client._ros = Mock()
        client._ts_mgr = Mock()
        client._connection_state = ConnectionState.CONNECTED
        
        client.terminate()
        
        assert client._ros is None
        assert client._ts_mgr is None
        assert client._connection_state == ConnectionState.DISCONNECTED

    def test_safe_publish_retries(self):
        """Test safe_publish with retries."""
        client = RosClient("ws://localhost:9090")
        client._ts_mgr = Mock()
        client._connection_state = ConnectionState.CONNECTED
        
        mock_topic = Mock()
        client._ts_mgr.topic.return_value = mock_topic
        
        # First attempt fails, second succeeds
        mock_topic.publish.side_effect = [Exception("Error"), None]
        
        client.safe_publish("/test/topic", "std_msgs/String", {"data": "test"}, retries=1)
        
        assert mock_topic.publish.call_count == 2

    def test_safe_publish_all_fail(self):
        """Test safe_publish when all attempts fail."""
        client = RosClient("ws://localhost:9090")
        client._ts_mgr = Mock()
        client._connection_state = ConnectionState.CONNECTED
        
        mock_topic = Mock()
        mock_topic.publish.side_effect = Exception("Error")
        client._ts_mgr.topic.return_value = mock_topic
        
        with pytest.raises(Exception):
            client.safe_publish("/test/topic", "std_msgs/String", {"data": "test"}, retries=1)

