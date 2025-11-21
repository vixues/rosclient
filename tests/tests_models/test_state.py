"""Tests for ConnectionState enum."""
import pytest

from rosclient.models.state import ConnectionState


class TestConnectionState:
    """Tests for ConnectionState enum."""

    def test_enum_values(self):
        """Test that all enum values are correct."""
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.DISCONNECTED.value == "disconnected"
        assert ConnectionState.ERROR.value == "error"
        assert ConnectionState.TIMEOUT.value == "timeout"
        assert ConnectionState.CLOSED.value == "closed"
        assert ConnectionState.RECONNECTING.value == "reconnecting"
        assert ConnectionState.RECONNECTED.value == "reconnected"

    def test_enum_comparison(self):
        """Test enum comparison."""
        state1 = ConnectionState.CONNECTED
        state2 = ConnectionState.CONNECTED
        state3 = ConnectionState.DISCONNECTED
        
        assert state1 == state2
        assert state1 != state3

    def test_enum_string_conversion(self):
        """Test converting enum to string."""
        state = ConnectionState.CONNECTED
        assert str(state) == "ConnectionState.CONNECTED"
        assert state.value == "connected"

    def test_all_states_exist(self):
        """Test that all expected states exist."""
        expected_states = {
            "CONNECTING",
            "CONNECTED",
            "DISCONNECTED",
            "ERROR",
            "TIMEOUT",
            "CLOSED",
            "RECONNECTING",
            "RECONNECTED",
        }
        actual_states = {state.name for state in ConnectionState}
        assert actual_states == expected_states

