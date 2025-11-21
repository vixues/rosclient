"""Tests for DroneState and RosTopic models."""
import time
import pytest

from rosclient.models.drone import DroneState, RosTopic
from rosclient.models.state import ConnectionState


class TestDroneState:
    """Tests for DroneState dataclass."""

    def test_default_initialization(self):
        """Test default initialization of DroneState."""
        state = DroneState()
        assert state.connected is False
        assert state.armed is False
        assert state.mode == ""
        assert state.battery == 100.0
        assert state.latitude == 0.0
        assert state.longitude == 0.0
        assert state.altitude == 0.0
        assert state.roll == 0.0
        assert state.pitch == 0.0
        assert state.yaw == 0.0
        assert state.landed is True
        assert state.reached is False
        assert state.returned is False
        assert state.tookoff is False
        assert isinstance(state.last_updated, float)
        assert state.last_updated > 0

    def test_custom_initialization(self):
        """Test custom initialization of DroneState."""
        state = DroneState(
            connected=True,
            armed=True,
            mode="GUIDED",
            battery=75.5,
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
        assert state.connected is True
        assert state.armed is True
        assert state.mode == "GUIDED"
        assert state.battery == 75.5
        assert state.latitude == 22.5329
        assert state.longitude == 113.93029
        assert state.altitude == 100.0
        assert state.roll == 1.5
        assert state.pitch == 2.3
        assert state.yaw == 45.0
        assert state.landed is False
        assert state.reached is True
        assert state.returned is False
        assert state.tookoff is True

    def test_state_mutation(self):
        """Test mutating DroneState fields."""
        state = DroneState()
        state.connected = True
        state.armed = True
        state.mode = "AUTO"
        state.battery = 50.0
        assert state.connected is True
        assert state.armed is True
        assert state.mode == "AUTO"
        assert state.battery == 50.0

    def test_last_updated_timestamp(self):
        """Test that last_updated is automatically set."""
        before = time.time()
        state = DroneState()
        after = time.time()
        assert before <= state.last_updated <= after


class TestRosTopic:
    """Tests for RosTopic dataclass."""

    def test_required_fields(self):
        """Test that name and type are required."""
        topic = RosTopic(name="/test/topic", type="std_msgs/String")
        assert topic.name == "/test/topic"
        assert topic.type == "std_msgs/String"
        assert topic.last_message is None
        assert isinstance(topic.last_message_time, float)
        assert topic.last_message_time > 0

    def test_with_message(self):
        """Test RosTopic with last_message."""
        message = {"data": "test"}
        topic = RosTopic(
            name="/test/topic",
            type="std_msgs/String",
            last_message=message,
        )
        assert topic.last_message == message
        assert topic.last_message["data"] == "test"

    def test_string_representation(self):
        """Test string representation of RosTopic."""
        topic = RosTopic(name="/test/topic", type="std_msgs/String")
        str_repr = str(topic)
        assert "Topic" in str_repr
        assert "/test/topic" in str_repr
        assert "std_msgs/String" in str_repr

    def test_repr_equals_str(self):
        """Test that __repr__ equals __str__."""
        topic = RosTopic(name="/test/topic", type="std_msgs/String")
        assert repr(topic) == str(topic)

    def test_post_init_zero_time(self):
        """Test __post_init__ handles zero time correctly."""
        topic = RosTopic(
            name="/test/topic",
            type="std_msgs/String",
            last_message_time=0.0,
        )
        assert topic.last_message_time > 0

