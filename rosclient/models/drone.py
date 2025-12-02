"""Drone state and topic data models."""
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


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
class Waypoint:
    """Waypoint for UAV navigation."""
    latitude: float
    longitude: float
    altitude: float
    id: int = 0
    name: str = ""
    action: str = "waypoint"  # waypoint, takeoff, land, etc.
    speed: float = 0.0  # Speed to reach this waypoint
    yaw: Optional[float] = None  # Optional yaw angle
    radius: float = 5.0  # Acceptance radius in meters
    hold_time: float = 0.0  # Time to hold at waypoint
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert waypoint to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
            "action": self.action,
            "speed": self.speed,
            "yaw": self.yaw,
            "radius": self.radius,
            "hold_time": self.hold_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Waypoint':
        """Create waypoint from dictionary."""
        return cls(
            id=data.get("id", 0),
            name=data.get("name", ""),
            latitude=float(data.get("latitude", 0.0)),
            longitude=float(data.get("longitude", 0.0)),
            altitude=float(data.get("altitude", 10.0)),
            action=data.get("action", "waypoint"),
            speed=float(data.get("speed", 0.0)),
            yaw=data.get("yaw"),
            radius=float(data.get("radius", 5.0)),
            hold_time=float(data.get("hold_time", 0.0))
        )


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

