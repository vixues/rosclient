"""Drone state and topic data models."""
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class DroneState:
    """Drone state information."""
    connected: bool = False
    armed: bool = False
    mode: str = ""
    battery: float = 100.0
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    landed: bool = True
    reached: bool = False
    returned: bool = False
    tookoff: bool = False
    last_updated: float = field(default_factory=time.time)


@dataclass
class RosTopic:
    """ROS topic information."""
    name: str
    type: str
    last_message: Optional[Dict[str, Any]] = None
    last_message_time: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Initialize last_message_time if not set."""
        if self.last_message_time == 0:
            self.last_message_time = time.time()

    def __str__(self) -> str:
        """String representation."""
        return f"Topic(name={self.name}, type={self.type}, last_message_time={self.last_message_time})"
    
    def __repr__(self) -> str:
        """Representation."""
        return self.__str__()

