"""Intelligent image processing module for ROS camera messages."""
from __future__ import annotations

import base64
import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple, Dict, Any, List, Callable

import cv2
import numpy as np


class ImageFormat(Enum):
    """Supported image output formats."""
    BGR = "bgr"  # OpenCV default
    RGB = "rgb"
    GRAY = "gray"
    HSV = "hsv"


class MessageType(Enum):
    """ROS message type detection."""
    UNKNOWN = "unknown"
    RAW_IMAGE = "sensor_msgs/Image"
    COMPRESSED_IMAGE = "sensor_msgs/CompressedImage"
    CUSTOM = "custom"


class ImageDecoder(ABC):
    """Base class for image decoders."""
    
    @abstractmethod
    def can_decode(self, msg: dict) -> bool:
        """Check if this decoder can handle the message."""
        pass
    
    @abstractmethod
    def decode(self, msg: dict) -> Optional[np.ndarray]:
        """Decode message to numpy array."""
        pass


class CompressedImageDecoder(ImageDecoder):
    """Decoder for sensor_msgs/CompressedImage."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.log = logger or logging.getLogger(self.__class__.__name__)
    
    def can_decode(self, msg: dict) -> bool:
        """Check for CompressedImage format."""
        return "format" in msg and "data" in msg
    
    def decode(self, msg: dict) -> Optional[np.ndarray]:
        """Decode compressed image."""
        try:
            data = msg.get("data")
            if not data:
                return None
            
            # Handle base64 string
            if isinstance(data, str):
                img_data = base64.b64decode(data)
            elif isinstance(data, (bytes, bytearray)):
                img_data = bytes(data)
            else:
                return None
            
            # Decode with OpenCV
            np_arr = np.frombuffer(img_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                # Try alternative decode flags
                frame = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                if frame is not None and len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            return frame
        except Exception as e:
            self.log.warning(f"CompressedImage decode failed: {e}")
            return None


class RawImageDecoder(ImageDecoder):
    """Decoder for sensor_msgs/Image (raw)."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.log = logger or logging.getLogger(self.__class__.__name__)
        # Encoding to channels mapping
        self._encoding_map = {
            "bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4,
            "mono8": 1, "8UC1": 1, "8UC3": 3, "8UC4": 4,
            "32FC1": 1, "32FC3": 3, "32FC4": 4,
        }
    
    def can_decode(self, msg: dict) -> bool:
        """Check for raw Image format."""
        return "encoding" in msg and "data" in msg
    
    def decode(self, msg: dict) -> Optional[np.ndarray]:
        """Decode raw image."""
        try:
            height = int(msg.get("height", 0))
            width = int(msg.get("width", 0))
            encoding = msg.get("encoding", "bgr8")
            data = msg.get("data")
            
            if height <= 0 or width <= 0 or not data:
                return None
            
            # Determine channels and dtype
            channels = self._encoding_map.get(encoding, 3)
            if encoding.startswith("32F"):
                dtype = np.float32
            else:
                dtype = np.uint8
            
            # Handle different data types
            if isinstance(data, str):
                img_data = base64.b64decode(data)
            elif isinstance(data, (bytes, bytearray)):
                img_data = bytes(data)
            else:
                return None
            
            np_arr = np.frombuffer(img_data, dtype=dtype)
            expected_size = height * width * channels
            
            if np_arr.size != expected_size:
                self.log.warning(f"Size mismatch: expected {expected_size}, got {np_arr.size}")
                return None
            
            # Reshape
            if channels == 1:
                frame = np_arr.reshape((height, width))
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif channels == 3:
                frame = np_arr.reshape((height, width, channels))
                if encoding == "rgb8":
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif channels == 4:
                frame = np_arr.reshape((height, width, channels))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:
                return None
            
            return frame
        except Exception as e:
            self.log.warning(f"RawImage decode failed: {e}")
            return None


class LegacyDecoder(ImageDecoder):
    """Legacy decoder for backward compatibility."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.log = logger or logging.getLogger(self.__class__.__name__)
    
    def can_decode(self, msg: dict) -> bool:
        """Fallback decoder."""
        return "data" in msg
    
    def decode(self, msg: dict) -> Optional[np.ndarray]:
        """Legacy decode logic."""
        try:
            data = msg.get("data")
            if isinstance(data, (bytes, bytearray)):
                np_arr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
            elif isinstance(data, str):
                img_data = base64.b64decode(data)
                np_arr = np.frombuffer(img_data, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
                
                # Fallback: manual reshape
                width = msg.get("width")
                height = msg.get("height")
                encoding = msg.get("encoding", "bgr8")
                channels = 3 if encoding in ["bgr8", "rgb8"] else 1
                
                if width and height:
                    frame = np_arr.reshape((height, width, channels))
                    if encoding == "rgb8":
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    return frame
            return None
        except Exception as e:
            self.log.warning(f"Legacy decode failed: {e}")
            return None


class ImagePostProcessor:
    """Post-processing pipeline for images."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize post-processor.
        
        Args:
            config: Processing configuration
                - output_format: ImageFormat enum
                - resize: (width, height) or None
                - keep_aspect: bool (for resize)
                - normalize: bool (0-1 range)
        """
        self.config = config or {}
        self._resize_cache: Optional[Tuple[int, int]] = None
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Apply post-processing pipeline."""
        if frame is None:
            return frame
        
        # Resize
        resize = self.config.get("resize")
        if resize:
            frame = self._resize(frame, resize, self.config.get("keep_aspect", True))
        
        # Format conversion
        output_format = self.config.get("output_format")
        if output_format:
            frame = self._convert_format(frame, output_format)
        
        # Normalize
        if self.config.get("normalize", False):
            frame = frame.astype(np.float32) / 255.0
        
        return frame
    
    def _resize(self, frame: np.ndarray, size: Tuple[int, int], keep_aspect: bool) -> np.ndarray:
        """Resize image."""
        h, w = frame.shape[:2]
        target_w, target_h = size
        
        if keep_aspect:
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        return frame
    
    def _convert_format(self, frame: np.ndarray, fmt: ImageFormat) -> np.ndarray:
        """Convert image format."""
        if fmt == ImageFormat.BGR:
            return frame
        elif fmt == ImageFormat.RGB:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif fmt == ImageFormat.GRAY:
            if len(frame.shape) == 3:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame
        elif fmt == ImageFormat.HSV:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return frame


class AlgorithmPlugin(ABC):
    """Base class for algorithm plugins (e.g., YOLO)."""
    
    @abstractmethod
    def process(self, image: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process image with algorithm.
        
        Args:
            image: Input image (BGR format)
            metadata: Optional metadata (timestamp, etc.)
            
        Returns:
            Processing results dictionary
        """
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if algorithm is ready."""
        pass


class ImageProcessor:
    """Intelligent image processor with plugin support."""
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize image processor.
        
        Args:
            logger: Optional logger instance
            config: Processing configuration
                - output_format: ImageFormat
                - resize: (width, height)
                - keep_aspect: bool
                - normalize: bool
                - enable_cache: bool
        """
        self.log = logger or logging.getLogger(self.__class__.__name__)
        self.config = config or {}
        
        # Initialize decoders (ordered by priority)
        self._decoders: List[ImageDecoder] = [
            CompressedImageDecoder(self.log),
            RawImageDecoder(self.log),
            LegacyDecoder(self.log),
        ]
        
        # Post-processor
        self._post_processor = ImagePostProcessor(self.config)
        
        # Algorithm plugins
        self._plugins: List[AlgorithmPlugin] = []
        
        # Performance cache
        self._last_result: Optional[Tuple[np.ndarray, float, Dict[str, Any]]] = None
        self._cache_enabled = self.config.get("enable_cache", True)
    
    def register_plugin(self, plugin: AlgorithmPlugin) -> None:
        """Register an algorithm plugin."""
        if plugin.is_ready():
            self._plugins.append(plugin)
            self.log.info(f"Registered plugin: {plugin.__class__.__name__}")
        else:
            self.log.warning(f"Plugin not ready: {plugin.__class__.__name__}")
    
    def unregister_plugin(self, plugin: AlgorithmPlugin) -> None:
        """Unregister an algorithm plugin."""
        if plugin in self._plugins:
            self._plugins.remove(plugin)
            self.log.info(f"Unregistered plugin: {plugin.__class__.__name__}")
    
    def detect_message_type(self, msg: dict) -> MessageType:
        """Detect ROS message type."""
        if "format" in msg and "data" in msg:
            return MessageType.COMPRESSED_IMAGE
        elif "encoding" in msg and "data" in msg:
            return MessageType.RAW_IMAGE
        return MessageType.UNKNOWN
    
    def decode_message(self, msg: dict) -> Optional[np.ndarray]:
        """
        Decode ROS message to OpenCV image.
        
        Args:
            msg: ROS camera message dictionary
            
        Returns:
            Decoded image array or None
        """
        # Try decoders in order
        for decoder in self._decoders:
            if decoder.can_decode(msg):
                frame = decoder.decode(msg)
                if frame is not None:
                    return frame
        
        self.log.warning("No decoder could handle the message")
        return None
    
    def process(
        self,
        msg: dict,
        apply_plugins: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[np.ndarray, float, Dict[str, Any]]]:
        """
        Process camera message and return image with metadata.
        
        Args:
            msg: ROS camera message dictionary
            apply_plugins: Whether to apply algorithm plugins
            metadata: Additional metadata to pass to plugins
            
        Returns:
            Tuple of (image, timestamp, results_dict) or None
            results_dict contains plugin outputs
        """
        # Decode
        frame = self.decode_message(msg)
        if frame is None:
            return None
        
        timestamp = time.time()
        
        # Post-processing
        frame = self._post_processor.process(frame)
        
        # Algorithm plugins
        results = {}
        segmented_image = None
        if apply_plugins and self._plugins:
            plugin_metadata = metadata or {}
            plugin_metadata["timestamp"] = timestamp
            plugin_metadata["shape"] = frame.shape
            
            for plugin in self._plugins:
                try:
                    plugin_result = plugin.process(frame, plugin_metadata)
                    results[plugin.__class__.__name__] = plugin_result
                    
                    # Extract segmented image from SAM3 plugin if available
                    if (plugin.__class__.__name__ == "SAM3Plugin" and 
                        "segmented_image" in plugin_result and
                        plugin_result.get("enabled", False)):
                        segmented_image = plugin_result["segmented_image"]
                except Exception as e:
                    self.log.error(f"Plugin {plugin.__class__.__name__} failed: {e}")
        
        # Use segmented image if available, otherwise use original
        output_image = segmented_image if segmented_image is not None else frame
        
        # Cache result
        if self._cache_enabled:
            self._last_result = (output_image, timestamp, results)
        
        return output_image, timestamp, results
    
    def process_simple(self, msg: dict) -> Optional[Tuple[np.ndarray, float]]:
        """
        Simple processing without plugins (backward compatibility).
        
        Returns:
            Tuple of (image, timestamp) or None
        """
        result = self.process(msg, apply_plugins=False)
        if result:
            frame, timestamp, _ = result
            return frame, timestamp
        return None
    
    def get_last_result(self) -> Optional[Tuple[np.ndarray, float, Dict[str, Any]]]:
        """Get cached last processing result."""
        return self._last_result
    
    def get_segmented_image(self) -> Optional[np.ndarray]:
        """
        Get the last segmented image from SAM3 plugin if available.
        
        Returns:
            Segmented image or None if not available
        """
        if self._last_result:
            _, _, results = self._last_result
            sam3_result = results.get("SAM3Plugin", {})
            if sam3_result.get("enabled", False) and "segmented_image" in sam3_result:
                return sam3_result["segmented_image"]
        return None
    
    def get_plugin_enabled(self, plugin_name: str) -> bool:
        """
        Check if a plugin is enabled.
        
        Args:
            plugin_name: Name of the plugin class
            
        Returns:
            True if plugin is enabled, False otherwise
        """
        for plugin in self._plugins:
            if plugin.__class__.__name__ == plugin_name:
                if hasattr(plugin, "enabled"):
                    return plugin.enabled
                return True
        return False
    
    def set_plugin_enabled(self, plugin_name: str, enabled: bool) -> bool:
        """
        Enable or disable a plugin.
        
        Args:
            plugin_name: Name of the plugin class
            enabled: Whether to enable the plugin
            
        Returns:
            True if plugin was found and updated, False otherwise
        """
        for plugin in self._plugins:
            if plugin.__class__.__name__ == plugin_name:
                if hasattr(plugin, "enable") and hasattr(plugin, "disable"):
                    if enabled:
                        plugin.enable()
                    else:
                        plugin.disable()
                    return True
        return False
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update processing configuration."""
        self.config.update(config)
        self._post_processor = ImagePostProcessor(self.config)
        self._cache_enabled = self.config.get("enable_cache", True)
