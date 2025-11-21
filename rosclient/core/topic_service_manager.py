"""Topic and Service manager for ROS connections."""
import logging
import threading
from typing import Dict

import roslibpy


class TopicServiceManager:
    """
    A simplified and standardized manager for roslibpy Topics and Services.
    Only uses the 'name' as the key. Does NOT manage topic/service types.
    """

    def __init__(self, ros: roslibpy.Ros, conn_id: str, logger_level: int = logging.DEBUG):
        """
        Initialize the TopicServiceManager.
        
        Args:
            ros: roslibpy.Ros instance
            conn_id: Connection identifier for logging
            logger_level: Logging level
        """
        self._ros = ros

        # Cache by name ONLY
        self._topics: Dict[str, roslibpy.Topic] = {}
        self._services: Dict[str, roslibpy.Service] = {}

        self._lock = threading.Lock()

        self.log = logging.getLogger(f"TopicService[{conn_id}]")
        self.log.setLevel(logger_level)

    # ----------------------------------------------------------------------
    # Topic
    # ----------------------------------------------------------------------
    def topic(self, name: str, ttype: str = "") -> roslibpy.Topic:
        """
        Get or create topic by name only. Type is optional and ignored for key.
        
        Args:
            name: Topic name
            ttype: Topic type (optional)
            
        Returns:
            roslibpy.Topic instance
        """
        with self._lock:
            if name not in self._topics:
                self._topics[name] = roslibpy.Topic(self._ros, name, ttype)
                self.log.debug(f"Created topic '{name}'")
            return self._topics[name]

    # ----------------------------------------------------------------------
    # Service
    # ----------------------------------------------------------------------
    def service(self, name: str, stype: str = "") -> roslibpy.Service:
        """
        Get or create service by name only. Type is optional and ignored for key.
        
        Args:
            name: Service name
            stype: Service type (optional)
            
        Returns:
            roslibpy.Service instance
        """
        with self._lock:
            if name not in self._services:
                self._services[name] = roslibpy.Service(self._ros, name, stype)
                self.log.debug(f"Created service '{name}'")
            return self._services[name]

    # ----------------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------------
    def close_all(self) -> None:
        """Close all topics and services."""
        with self._lock:
            for name, t in list(self._topics.items()):
                try:
                    t.unsubscribe()
                    self.log.info(f"Unsubscribed topic '{name}'")
                except Exception as e:
                    self.log.warning(f"Failed to unsubscribe topic '{name}': {e}")

            for name, s in list(self._services.items()):
                try:
                    if hasattr(s, "unadvertise"):
                        s.unadvertise()
                        self.log.info(f"Unadvertised service '{name}'")
                except Exception as e:
                    self.log.warning(f"Failed to unadvertise service '{name}': {e}")

            self._topics.clear()
            self._services.clear()
            self.log.info("Cleared all topics and services.")

