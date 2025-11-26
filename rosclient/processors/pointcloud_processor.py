"""Point cloud processing utilities for ROS PointCloud2 messages."""
from __future__ import annotations

import base64
import logging
import time
from typing import Optional, Tuple, List

import numpy as np


class PointCloudProcessor:
    """Process ROS PointCloud2 messages into numpy arrays."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize point cloud processor.
        
        Args:
            logger: Optional logger instance
        """
        self.log = logger or logging.getLogger(self.__class__.__name__)
    
    def decode_message(self, msg: dict) -> Optional[np.ndarray]:
        """
        Decode ROS PointCloud2 message to numpy array.
        
        Args:
            msg: PointCloud2 message dictionary
            
        Returns:
            Array of points (N, 3) or None if decoding fails
        """
        try:
            if "data" not in msg or "fields" not in msg:
                return None

            raw = msg["data"]
            # Handle base64 or raw bytes
            if isinstance(raw, str):
                raw_data = base64.b64decode(raw)
            elif isinstance(raw, (bytes, bytearray)):
                raw_data = bytes(raw)
            else:
                return None

            np_data = np.frombuffer(raw_data, dtype=np.uint8)
            fields = msg["fields"]
            point_step = int(msg.get("point_step", 0))
            
            if point_step <= 0:
                return None

            # Find field offsets
            x_offset = next((f["offset"] for f in fields if f["name"] == "x"), None)
            y_offset = next((f["offset"] for f in fields if f["name"] == "y"), None)
            z_offset = next((f["offset"] for f in fields if f["name"] == "z"), None)
            
            if None in (x_offset, y_offset, z_offset):
                return None

            # Extract points
            points: List[Tuple[float, float, float]] = []
            total_len = len(np_data)
            
            for i in range(0, total_len - point_step + 1, point_step):
                try:
                    x = np.frombuffer(np_data[i + x_offset:i + x_offset + 4], dtype=np.float32)[0]
                    y = np.frombuffer(np_data[i + y_offset:i + y_offset + 4], dtype=np.float32)[0]
                    z = np.frombuffer(np_data[i + z_offset:i + z_offset + 4], dtype=np.float32)[0]
                    points.append((x, y, z))
                except Exception:
                    continue

            if not points:
                return None

            return np.array(points, dtype=np.float32)
        except Exception as e:
            self.log.error(f"Point cloud decode error: {e}")
            return None
    
    def process(self, msg: dict) -> Optional[Tuple[np.ndarray, float]]:
        """
        Process PointCloud2 message and return points with timestamp.
        
        Args:
            msg: PointCloud2 message dictionary
            
        Returns:
            Tuple of (points array, timestamp) or None
        """
        points = self.decode_message(msg)
        if points is None:
            return None
        return points, time.time()

