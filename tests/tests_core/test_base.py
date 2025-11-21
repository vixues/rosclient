"""Tests for RosClientBase."""
import pytest
from unittest.mock import Mock, patch
import numpy as np

from rosclient.core.base import RosClientBase
from rosclient.models.state import ConnectionState
from rosclient.clients.mock_client import MockRosClient


class TestRosClientBase:
    """Tests for RosClientBase abstract class."""

    def test_initialization(self):
        """Test base class initialization."""
        client = MockRosClient("ws://localhost:9090")
        assert client.connection_str == "ws://localhost:9090"
        assert isinstance(client._config, dict)
        assert client._connection_state == ConnectionState.CONNECTED  # Mock starts connected

    def test_is_connected(self):
        """Test is_connected method."""
        client = MockRosClient("ws://localhost:9090")
        assert client.is_connected() is True
        
        client.terminate()
        assert client.is_connected() is False

    def test_get_status(self):
        """Test get_status method."""
        client = MockRosClient("ws://localhost:9090")
        state = client.get_status()
        assert state.connected is True
        assert state.battery == 100.0

    def test_get_position(self):
        """Test get_position method."""
        client = MockRosClient("ws://localhost:9090")
        pos = client.get_position()
        assert len(pos) == 3
        assert isinstance(pos[0], float)  # latitude
        assert isinstance(pos[1], float)  # longitude
        assert isinstance(pos[2], float)  # altitude

    def test_get_orientation(self):
        """Test get_orientation method."""
        client = MockRosClient("ws://localhost:9090")
        ori = client.get_orientation()
        assert len(ori) == 3
        assert isinstance(ori[0], float)  # roll
        assert isinstance(ori[1], float)  # pitch
        assert isinstance(ori[2], float)  # yaw

    def test_decode_point_cloud_valid(self):
        """Test _decode_point_cloud with valid data."""
        client = MockRosClient("ws://localhost:9090")
        
        # Create a mock point cloud message
        msg = {
            "data": b"\x00" * 48,  # 3 points * 16 bytes each
            "fields": [
                {"name": "x", "offset": 0},
                {"name": "y", "offset": 4},
                {"name": "z", "offset": 8},
            ],
            "point_step": 16,
        }
        
        # Fill with some float data
        import struct
        points_data = b""
        for i in range(3):
            points_data += struct.pack("fff", float(i), float(i+1), float(i+2))
            points_data += b"\x00" * 4  # padding
        
        msg["data"] = points_data
        
        result = client._decode_point_cloud(msg)
        assert result is not None
        points, timestamp = result
        assert isinstance(points, np.ndarray)
        assert len(points) == 3

    def test_decode_point_cloud_invalid(self):
        """Test _decode_point_cloud with invalid data."""
        client = MockRosClient("ws://localhost:9090")
        
        # Missing fields
        msg = {"data": b"test"}
        result = client._decode_point_cloud(msg)
        assert result is None
        
        # Missing data
        msg = {"fields": []}
        result = client._decode_point_cloud(msg)
        assert result is None

    def test_update_odom(self):
        """Test update_odom method."""
        client = MockRosClient("ws://localhost:9090")
        
        msg = {
            "pose": {
                "pose": {
                    "orientation": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                        "w": 1.0,
                    },
                    "position": {
                        "x": 1.0,
                        "y": 2.0,
                        "z": 3.0,
                    },
                }
            }
        }
        
        client.update_odom(msg)
        state = client.get_status()
        assert state.roll == 0.0
        assert state.pitch == 0.0
        assert state.yaw == 0.0

    def test_update_odom_invalid(self):
        """Test update_odom with invalid data."""
        client = MockRosClient("ws://localhost:9090")
        initial_state = client.get_status()
        
        # Invalid message
        msg = {}
        client.update_odom(msg)
        
        # State should still be valid (handles exception gracefully)
        state = client.get_status()
        assert state is not None

