"""High-performance recording module for ROS client data."""
from __future__ import annotations

import gzip
import json
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Callable

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False
    msgpack = None

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None

from ..models.drone import DroneState
from ..utils.logger import setup_logger


@dataclass
class RecordEntry:
    """Single record entry with timestamp."""
    timestamp: float
    data_type: str  # 'image', 'pointcloud', 'state'
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecordMetadata:
    """Metadata for a recording session."""
    start_time: float
    end_time: float
    total_duration: float
    image_count: int = 0
    pointcloud_count: int = 0
    state_count: int = 0
    version: str = "1.0"
    client_type: str = ""
    connection_str: str = ""
    config: Dict[str, Any] = field(default_factory=dict)


class Recorder:
    """
    High-performance recorder for ROS client data.
    
    Features:
    - Asynchronous recording with background thread
    - Efficient compression for images and point clouds
    - Thread-safe operations
    - Support for image, point cloud, and state recording
    """
    
    def __init__(
        self,
        record_images: bool = True,
        record_pointclouds: bool = True,
        record_states: bool = True,
        image_quality: int = 85,
        max_queue_size: int = 100,
        batch_size: int = 10,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the recorder.
        
        Args:
            record_images: Whether to record images
            record_pointclouds: Whether to record point clouds
            record_states: Whether to record states
            image_quality: JPEG quality (1-100) for image compression
            max_queue_size: Maximum queue size for buffering
            batch_size: Number of entries to batch before writing
            logger: Optional logger instance
        """
        self.record_images = record_images
        self.record_pointclouds = record_pointclouds
        self.record_states = record_states
        self.image_quality = max(1, min(100, image_quality))
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        
        self.log = logger or setup_logger("Recorder")
        
        # Recording state
        self._is_recording = False
        self._lock = threading.RLock()
        self._record_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._write_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Recorded data (in-memory for fast access)
        self._recorded_entries: List[RecordEntry] = []
        self._metadata = RecordMetadata(
            start_time=0.0,
            end_time=0.0,
            total_duration=0.0
        )
        
        # Statistics
        self._stats = {
            "images_recorded": 0,
            "pointclouds_recorded": 0,
            "states_recorded": 0,
            "dropped": 0
        }
        
    def start_recording(self, client_type: str = "", connection_str: str = "", config: Dict[str, Any] = None) -> None:
        """
        Start recording.
        
        Args:
            client_type: Type of client (e.g., "RosClient", "MockRosClient")
            connection_str: Connection string
            config: Client configuration
        """
        with self._lock:
            if self._is_recording:
                self.log.warning("Recording already in progress")
                return
            
            self._is_recording = True
            self._stop_event.clear()
            self._recorded_entries.clear()
            self._metadata = RecordMetadata(
                start_time=time.time(),
                end_time=0.0,
                total_duration=0.0,
                client_type=client_type,
                connection_str=connection_str,
                config=config or {}
            )
            self._stats = {
                "images_recorded": 0,
                "pointclouds_recorded": 0,
                "states_recorded": 0,
                "dropped": 0
            }
            
            # Start background writer thread
            self._write_thread = threading.Thread(
                target=self._write_worker,
                daemon=True,
                name="Recorder-Writer"
            )
            self._write_thread.start()
            self.log.info("Recording started")
    
    def stop_recording(self) -> None:
        """Stop recording and flush all pending data."""
        with self._lock:
            if not self._is_recording:
                return
            
            self._is_recording = False
            self._stop_event.set()
            
            # Wait for writer thread to finish
            if self._write_thread and self._write_thread.is_alive():
                self._write_thread.join(timeout=5.0)
            
            self._metadata.end_time = time.time()
            self._metadata.total_duration = self._metadata.end_time - self._metadata.start_time
            self._metadata.image_count = self._stats["images_recorded"]
            self._metadata.pointcloud_count = self._stats["pointclouds_recorded"]
            self._metadata.state_count = self._stats["states_recorded"]
            
            self.log.info(
                f"Recording stopped. Duration: {self._metadata.total_duration:.2f}s, "
                f"Images: {self._stats['images_recorded']}, "
                f"PointClouds: {self._stats['pointclouds_recorded']}, "
                f"States: {self._stats['states_recorded']}"
            )
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        with self._lock:
            return self._is_recording
    
    def record_image(self, image: np.ndarray, timestamp: float, state: Optional[DroneState] = None) -> bool:
        """
        Record an image with optional synchronized state.
        
        Args:
            image: Image array (H, W, 3) RGB format
            timestamp: Timestamp
            state: Optional synchronized state snapshot
            
        Returns:
            True if successfully queued, False otherwise
        """
        if not self.record_images or not self._is_recording:
            return False
        
        if not HAS_NUMPY or image is None:
            return False
        
        try:
            # Compress image to JPEG
            if HAS_CV2:
                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if len(image.shape) == 3 else image
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.image_quality]
                success, encoded_img = cv2.imencode('.jpg', img_bgr, encode_param)
                if success:
                    compressed_data = encoded_img.tobytes()
                else:
                    return False
            else:
                # Fallback: use numpy array directly (less efficient)
                compressed_data = image.tobytes()
            
            # Prepare metadata with optional state
            metadata = {
                "shape": image.shape,
                "dtype": str(image.dtype),
                "compressed": HAS_CV2
            }
            
            # Add state snapshot to metadata if provided
            if state is not None:
                from dataclasses import asdict
                metadata["synchronized_state"] = asdict(state)
            
            entry = RecordEntry(
                timestamp=timestamp,
                data_type="image",
                data=compressed_data,
                metadata=metadata
            )
            
            try:
                self._record_queue.put_nowait(entry)
                with self._lock:
                    self._stats["images_recorded"] += 1
                return True
            except queue.Full:
                with self._lock:
                    self._stats["dropped"] += 1
                self.log.warning("Record queue full, dropping image")
                return False
        except Exception as e:
            self.log.error(f"Error recording image: {e}")
            return False
    
    def record_pointcloud(self, points: np.ndarray, timestamp: float, state: Optional[DroneState] = None) -> bool:
        """
        Record a point cloud with optional synchronized state.
        
        Args:
            points: Point cloud array (N, 3)
            timestamp: Timestamp
            state: Optional synchronized state snapshot
            
        Returns:
            True if successfully queued, False otherwise
        """
        if not self.record_pointclouds or not self._is_recording:
            return False
        
        if not HAS_NUMPY or points is None or len(points) == 0:
            return False
        
        try:
            # Compress point cloud using numpy compression
            # Convert to float32 to save space
            points_float32 = points.astype(np.float32)
            compressed_data = gzip.compress(points_float32.tobytes())
            
            # Prepare metadata with optional state
            metadata = {
                "shape": points.shape,
                "dtype": str(points.dtype),
                "num_points": len(points),
                "compressed": True
            }
            
            # Add state snapshot to metadata if provided
            if state is not None:
                from dataclasses import asdict
                metadata["synchronized_state"] = asdict(state)
            
            entry = RecordEntry(
                timestamp=timestamp,
                data_type="pointcloud",
                data=compressed_data,
                metadata=metadata
            )
            
            try:
                self._record_queue.put_nowait(entry)
                with self._lock:
                    self._stats["pointclouds_recorded"] += 1
                return True
            except queue.Full:
                with self._lock:
                    self._stats["dropped"] += 1
                self.log.warning("Record queue full, dropping point cloud")
                return False
        except Exception as e:
            self.log.error(f"Error recording point cloud: {e}")
            return False
    
    def record_state(self, state: DroneState, timestamp: float) -> bool:
        """
        Record a drone state.
        
        Args:
            state: DroneState instance
            timestamp: Timestamp
            
        Returns:
            True if successfully queued, False otherwise
        """
        if not self.record_states or not self._is_recording:
            return False
        
        if state is None:
            return False
        
        try:
            # Convert state to dictionary
            state_dict = asdict(state)
            
            entry = RecordEntry(
                timestamp=timestamp,
                data_type="state",
                data=state_dict,
                metadata={}
            )
            
            try:
                self._record_queue.put_nowait(entry)
                with self._lock:
                    self._stats["states_recorded"] += 1
                return True
            except queue.Full:
                with self._lock:
                    self._stats["dropped"] += 1
                self.log.warning("Record queue full, dropping state")
                return False
        except Exception as e:
            self.log.error(f"Error recording state: {e}")
            return False
    
    def _write_worker(self) -> None:
        """Background worker thread that processes the record queue."""
        batch = []
        last_flush = time.time()
        
        while not self._stop_event.is_set() or not self._record_queue.empty():
            try:
                # Get entry with timeout
                try:
                    entry = self._record_queue.get(timeout=0.1)
                except queue.Empty:
                    # Flush batch if it's been a while
                    if batch and (time.time() - last_flush > 1.0):
                        self._flush_batch(batch)
                        batch.clear()
                        last_flush = time.time()
                    continue
                
                batch.append(entry)
                
                # Flush batch when it reaches batch_size
                if len(batch) >= self.batch_size:
                    self._flush_batch(batch)
                    batch.clear()
                    last_flush = time.time()
                    
            except Exception as e:
                self.log.error(f"Error in write worker: {e}")
        
        # Flush remaining batch
        if batch:
            self._flush_batch(batch)
    
    def _flush_batch(self, batch: List[RecordEntry]) -> None:
        """Flush a batch of entries to memory."""
        with self._lock:
            self._recorded_entries.extend(batch)
            # Sort by timestamp to maintain chronological order
            self._recorded_entries.sort(key=lambda x: x.timestamp)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get recording statistics."""
        with self._lock:
            return {
                **self._stats,
                "total_entries": len(self._recorded_entries),
                "queue_size": self._record_queue.qsize(),
                "is_recording": self._is_recording
            }
    
    def save(self, file_path: str, compress: bool = True) -> bool:
        """
        Save recorded data to file.
        
        Args:
            file_path: Path to save file
            compress: Whether to compress the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.stop_recording()  # Ensure recording is stopped
            
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for serialization
            data = {
                "metadata": asdict(self._metadata),
                "entries": []
            }
            
            # Serialize entries
            for entry in self._recorded_entries:
                entry_dict = {
                    "timestamp": entry.timestamp,
                    "data_type": entry.data_type,
                    "metadata": entry.metadata
                }
                
                # Handle different data types
                if entry.data_type == "image":
                    # Image is already compressed bytes
                    if isinstance(entry.data, bytes):
                        entry_dict["data"] = entry.data
                    else:
                        entry_dict["data"] = entry.data.tobytes() if HAS_NUMPY else str(entry.data)
                elif entry.data_type == "pointcloud":
                    # Point cloud is already compressed bytes
                    entry_dict["data"] = entry.data
                elif entry.data_type == "state":
                    # State is a dictionary
                    entry_dict["data"] = entry.data
                
                data["entries"].append(entry_dict)
            
            # Save using msgpack if available, otherwise JSON
            if HAS_MSGPACK and compress:
                # Use msgpack for binary data
                packed = msgpack.packb(data, use_bin_type=True)
                if compress:
                    packed = gzip.compress(packed)
                with open(path, 'wb') as f:
                    f.write(packed)
                self.log.info(f"Saved recording to {path} (msgpack, compressed)")
            else:
                # Fallback to JSON (note: binary data will be base64 encoded)
                if compress:
                    # For JSON, we need to base64 encode binary data
                    import base64
                    for entry in data["entries"]:
                        if entry["data_type"] in ["image", "pointcloud"]:
                            if isinstance(entry["data"], bytes):
                                entry["data"] = base64.b64encode(entry["data"]).decode('utf-8')
                    
                    json_str = json.dumps(data, indent=2)
                    compressed = gzip.compress(json_str.encode('utf-8'))
                    with open(path, 'wb') as f:
                        f.write(compressed)
                    self.log.info(f"Saved recording to {path} (JSON, compressed)")
                else:
                    # JSON without compression (not recommended for binary data)
                    import base64
                    for entry in data["entries"]:
                        if entry["data_type"] in ["image", "pointcloud"]:
                            if isinstance(entry["data"], bytes):
                                entry["data"] = base64.b64encode(entry["data"]).decode('utf-8')
                    
                    with open(path, 'w') as f:
                        json.dump(data, f, indent=2)
                    self.log.info(f"Saved recording to {path} (JSON, uncompressed)")
            
            return True
        except Exception as e:
            self.log.error(f"Error saving recording: {e}")
            return False
    
    @classmethod
    def load(cls, file_path: str, logger: Optional[logging.Logger] = None) -> Optional[Recorder]:
        """
        Load recorded data from file.
        
        Args:
            file_path: Path to load file
            logger: Optional logger instance
            
        Returns:
            Recorder instance with loaded data, or None if failed
        """
        try:
            log = logger or setup_logger("Recorder")
            path = Path(file_path)
            
            if not path.exists():
                log.error(f"Recording file not found: {path}")
                return None
            
            # Try to detect format and load
            data = None
            with open(path, 'rb') as f:
                raw_data = f.read()
            
            # Try msgpack first
            if HAS_MSGPACK:
                try:
                    # Try decompressing
                    try:
                        raw_data = gzip.decompress(raw_data)
                    except:
                        pass  # Not compressed
                    data = msgpack.unpackb(raw_data, raw=False)
                    log.info(f"Loaded recording from {path} (msgpack format)")
                except:
                    pass
            
            # Fallback to JSON
            if data is None:
                try:
                    # Try decompressing
                    try:
                        raw_data = gzip.decompress(raw_data)
                    except:
                        pass  # Not compressed
                    data = json.loads(raw_data.decode('utf-8'))
                    log.info(f"Loaded recording from {path} (JSON format)")
                except Exception as e:
                    log.error(f"Failed to load recording: {e}")
                    return None
            
            # Create recorder and populate data
            recorder = Recorder(logger=log)
            recorder._metadata = RecordMetadata(**data["metadata"])
            
            # Deserialize entries
            for entry_dict in data["entries"]:
                entry = RecordEntry(
                    timestamp=entry_dict["timestamp"],
                    data_type=entry_dict["data_type"],
                    data=None,  # Will be set below
                    metadata=entry_dict.get("metadata", {})
                )
                
                # Handle different data types
                if entry_dict["data_type"] == "image":
                    if isinstance(entry_dict["data"], str):
                        # Base64 encoded (from JSON)
                        import base64
                        entry.data = base64.b64decode(entry_dict["data"])
                    else:
                        entry.data = entry_dict["data"]
                elif entry_dict["data_type"] == "pointcloud":
                    if isinstance(entry_dict["data"], str):
                        # Base64 encoded (from JSON)
                        import base64
                        entry.data = base64.b64decode(entry_dict["data"])
                    else:
                        entry.data = entry_dict["data"]
                elif entry_dict["data_type"] == "state":
                    entry.data = entry_dict["data"]
                
                recorder._recorded_entries.append(entry)
            
            # Update statistics
            recorder._stats = {
                "images_recorded": recorder._metadata.image_count,
                "pointclouds_recorded": recorder._metadata.pointcloud_count,
                "states_recorded": recorder._metadata.state_count,
                "dropped": 0
            }
            
            log.info(
                f"Loaded recording: {len(recorder._recorded_entries)} entries, "
                f"Duration: {recorder._metadata.total_duration:.2f}s"
            )
            
            return recorder
        except Exception as e:
            log = logger or setup_logger("Recorder")
            log.error(f"Error loading recording: {e}")
            return None
    
    def get_entries(
        self,
        data_type: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[RecordEntry]:
        """
        Get recorded entries with optional filtering.
        
        Args:
            data_type: Filter by data type ('image', 'pointcloud', 'state')
            start_time: Start timestamp filter
            end_time: End timestamp filter
            
        Returns:
            List of matching entries
        """
        with self._lock:
            entries = self._recorded_entries
            
            if data_type:
                entries = [e for e in entries if e.data_type == data_type]
            
            if start_time is not None:
                entries = [e for e in entries if e.timestamp >= start_time]
            
            if end_time is not None:
                entries = [e for e in entries if e.timestamp <= end_time]
            
            return entries
    
    def decode_entry(self, entry: RecordEntry) -> Optional[Tuple[Any, float, Optional[DroneState]]]:
        """
        Decode a record entry back to original format with optional synchronized state.
        
        Args:
            entry: RecordEntry to decode
            
        Returns:
            Tuple of (data, timestamp, state) or None if decoding fails
            state is None if no synchronized state is available
        """
        try:
            state = None
            # Extract synchronized state from metadata if available
            if "synchronized_state" in entry.metadata:
                from ..models.drone import DroneState
                try:
                    state = DroneState(**entry.metadata["synchronized_state"])
                except Exception as e:
                    self.log.warning(f"Failed to decode synchronized state: {e}")
            
            if entry.data_type == "image":
                if HAS_CV2 and entry.metadata.get("compressed", False):
                    # Decode JPEG
                    img_array = np.frombuffer(entry.data, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is not None:
                        # Convert BGR to RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        return (img, entry.timestamp, state)
                else:
                    # Fallback: try to reconstruct from bytes
                    if HAS_NUMPY and "shape" in entry.metadata:
                        shape = tuple(entry.metadata["shape"])
                        dtype_str = entry.metadata.get("dtype", "uint8")
                        dtype = np.dtype(dtype_str)
                        img = np.frombuffer(entry.data, dtype=dtype).reshape(shape)
                        return (img, entry.timestamp, state)
            
            elif entry.data_type == "pointcloud":
                if HAS_NUMPY:
                    # Decompress point cloud
                    decompressed = gzip.decompress(entry.data)
                    shape = tuple(entry.metadata["shape"])
                    dtype_str = entry.metadata.get("dtype", "float32")
                    dtype = np.dtype(dtype_str)
                    points = np.frombuffer(decompressed, dtype=dtype).reshape(shape)
                    return (points, entry.timestamp, state)
            
            elif entry.data_type == "state":
                # State is already a dictionary
                from ..models.drone import DroneState
                state = DroneState(**entry.data)
                return (state, entry.timestamp, None)
            
            return None
        except Exception as e:
            self.log.error(f"Error decoding entry: {e}")
            return None

