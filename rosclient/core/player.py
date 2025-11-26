"""Record player for replaying recorded ROS client data."""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Dict, Any, Tuple, Callable, List

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

from .recorder import Recorder, RecordEntry
from ..models.drone import DroneState
from ..utils.logger import setup_logger


class RecordPlayer:
    """
    High-performance player for recorded ROS client data.
    
    Features:
    - Time-synchronized playback
    - Variable playback speed
    - Callback support for real-time playback
    - Support for seeking to specific timestamps
    """
    
    def __init__(
        self,
        recorder: Recorder,
        playback_speed: float = 1.0,
        loop: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the record player.
        
        Args:
            recorder: Recorder instance with loaded data
            playback_speed: Playback speed multiplier (1.0 = real-time, 2.0 = 2x speed)
            loop: Whether to loop playback
            logger: Optional logger instance
        """
        self.recorder = recorder
        self.playback_speed = max(0.1, min(10.0, playback_speed))  # Clamp between 0.1x and 10x
        self.loop = loop
        self.log = logger or setup_logger("RecordPlayer")
        
        # Playback state
        self._is_playing = False
        self._is_paused = False
        self._lock = threading.RLock()
        self._play_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Current position
        self._current_index = 0
        self._start_time: Optional[float] = None
        self._playback_start_time: Optional[float] = None
        
        # Callbacks
        self._image_callback: Optional[Callable[[np.ndarray, float], None]] = None
        self._pointcloud_callback: Optional[Callable[[np.ndarray, float], None]] = None
        self._state_callback: Optional[Callable[[DroneState, float], None]] = None
        
        # Statistics
        self._stats = {
            "entries_played": 0,
            "images_played": 0,
            "pointclouds_played": 0,
            "states_played": 0
        }
    
    def set_image_callback(self, callback: Callable[[np.ndarray, float], None]) -> None:
        """Set callback for image playback."""
        self._image_callback = callback
    
    def set_pointcloud_callback(self, callback: Callable[[np.ndarray, float], None]) -> None:
        """Set callback for point cloud playback."""
        self._pointcloud_callback = callback
    
    def set_state_callback(self, callback: Callable[[DroneState, float], None]) -> None:
        """Set callback for state playback."""
        self._state_callback = callback
    
    def play(self, start_time: Optional[float] = None) -> None:
        """
        Start playback.
        
        Args:
            start_time: Optional start timestamp (relative to recording start)
        """
        with self._lock:
            if self._is_playing:
                self.log.warning("Playback already in progress")
                return
            
            entries = self.recorder.get_entries()
            if not entries:
                self.log.error("No entries to play")
                return
            
            self._is_playing = True
            self._is_paused = False
            self._stop_event.clear()
            
            # Determine start index
            if start_time is not None:
                # Find entry closest to start_time
                recording_start = self.recorder._metadata.start_time
                target_time = recording_start + start_time
                self._current_index = self._find_index_for_time(target_time)
            else:
                self._current_index = 0
            
            self._start_time = entries[self._current_index].timestamp if entries else None
            self._playback_start_time = time.time()
            
            # Reset statistics
            self._stats = {
                "entries_played": 0,
                "images_played": 0,
                "pointclouds_played": 0,
                "states_played": 0
            }
            
            # Start playback thread
            self._play_thread = threading.Thread(
                target=self._play_worker,
                daemon=True,
                name="RecordPlayer-Worker"
            )
            self._play_thread.start()
            self.log.info(f"Playback started from index {self._current_index}")
    
    def pause(self) -> None:
        """Pause playback."""
        with self._lock:
            if not self._is_playing:
                return
            self._is_paused = True
            self.log.info("Playback paused")
    
    def resume(self) -> None:
        """Resume playback."""
        with self._lock:
            if not self._is_playing:
                return
            self._is_paused = False
            # Adjust playback start time to account for pause duration
            if self._playback_start_time:
                pause_duration = time.time() - (getattr(self, '_pause_start_time', time.time()))
                self._playback_start_time += pause_duration
            self.log.info("Playback resumed")
    
    def stop(self) -> None:
        """Stop playback."""
        with self._lock:
            if not self._is_playing:
                return
            
            self._is_playing = False
            self._is_paused = False
            self._stop_event.set()
            
            # Wait for play thread to finish
            if self._play_thread and self._play_thread.is_alive():
                self._play_thread.join(timeout=2.0)
            
            self.log.info(
                f"Playback stopped. Played {self._stats['entries_played']} entries "
                f"(Images: {self._stats['images_played']}, "
                f"PointClouds: {self._stats['pointclouds_played']}, "
                f"States: {self._stats['states_played']})"
            )
    
    def seek(self, timestamp: float) -> bool:
        """
        Seek to a specific timestamp (relative to recording start).
        
        Args:
            timestamp: Timestamp to seek to (seconds from recording start)
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            if self._is_playing:
                self.log.warning("Cannot seek while playing. Stop playback first.")
                return False
            
            recording_start = self.recorder._metadata.start_time
            target_time = recording_start + timestamp
            
            self._current_index = self._find_index_for_time(target_time)
            self.log.info(f"Seeked to index {self._current_index} (timestamp: {timestamp:.2f}s)")
            return True
    
    def _find_index_for_time(self, target_time: float) -> int:
        """Find the index of the entry closest to target_time."""
        entries = self.recorder.get_entries()
        if not entries:
            return 0
        
        # Binary search for closest entry
        left, right = 0, len(entries) - 1
        while left < right:
            mid = (left + right) // 2
            if entries[mid].timestamp < target_time:
                left = mid + 1
            else:
                right = mid
        
        return left
    
    def _play_worker(self) -> None:
        """Background worker thread that plays entries."""
        entries = self.recorder.get_entries()
        if not entries:
            return
        
        recording_start = self.recorder._metadata.start_time
        
        while not self._stop_event.is_set() and self._current_index < len(entries):
            # Handle pause
            if self._is_paused:
                time.sleep(0.1)
                continue
            
            entry = entries[self._current_index]
            
            # Calculate when this entry should be played
            relative_time = entry.timestamp - recording_start
            if self._start_time:
                relative_time -= (self._start_time - recording_start)
            
            # Wait until it's time to play this entry
            if self._playback_start_time:
                elapsed = time.time() - self._playback_start_time
                target_elapsed = relative_time / self.playback_speed
                
                if elapsed < target_elapsed:
                    sleep_time = (target_elapsed - elapsed) / self.playback_speed
                    time.sleep(max(0, min(sleep_time, 0.1)))  # Cap sleep time
            
            # Decode and play entry
            result = self.recorder.decode_entry(entry)
            if result:
                data, timestamp, synced_state = result
                self._play_entry(entry.data_type, data, timestamp, synced_state)
                
                with self._lock:
                    self._stats["entries_played"] += 1
                    if entry.data_type == "image":
                        self._stats["images_played"] += 1
                    elif entry.data_type == "pointcloud":
                        self._stats["pointclouds_played"] += 1
                    elif entry.data_type == "state":
                        self._stats["states_played"] += 1
            
            self._current_index += 1
            
            # Handle loop
            if self._current_index >= len(entries) and self.loop:
                self._current_index = 0
                self._playback_start_time = time.time()
                self._start_time = entries[0].timestamp if entries else None
                self.log.info("Playback looped to start")
        
        # Playback finished
        with self._lock:
            self._is_playing = False
    
    def _play_entry(
        self,
        data_type: str,
        data: Any,
        timestamp: float,
        synced_state: Optional[DroneState] = None
    ) -> None:
        """
        Play a single entry by calling appropriate callback.
        
        Args:
            data_type: Type of data ('image', 'pointcloud', 'state')
            data: The data to play
            timestamp: Timestamp of the data
            synced_state: Optional synchronized state snapshot
        """
        try:
            if data_type == "image" and self._image_callback:
                self._image_callback(data, timestamp)
                # If synchronized state is available, also update state callback
                if synced_state and self._state_callback:
                    self._state_callback(synced_state, timestamp)
            elif data_type == "pointcloud" and self._pointcloud_callback:
                self._pointcloud_callback(data, timestamp)
                # If synchronized state is available, also update state callback
                if synced_state and self._state_callback:
                    self._state_callback(synced_state, timestamp)
            elif data_type == "state" and self._state_callback:
                self._state_callback(data, timestamp)
        except Exception as e:
            self.log.error(f"Error in playback callback: {e}")
    
    def is_playing(self) -> bool:
        """Check if currently playing."""
        with self._lock:
            return self._is_playing
    
    def is_paused(self) -> bool:
        """Check if currently paused."""
        with self._lock:
            return self._is_paused and self._is_playing
    
    def get_current_time(self) -> float:
        """Get current playback time (relative to recording start)."""
        with self._lock:
            if not self._is_playing or not self._start_time:
                return 0.0
            
            entries = self.recorder.get_entries()
            if self._current_index >= len(entries):
                return self.recorder._metadata.total_duration
            
            current_entry = entries[min(self._current_index, len(entries) - 1)]
            recording_start = self.recorder._metadata.start_time
            return current_entry.timestamp - recording_start
    
    def get_progress(self) -> float:
        """
        Get playback progress as a fraction (0.0 to 1.0).
        
        Returns:
            Progress fraction
        """
        with self._lock:
            entries = self.recorder.get_entries()
            if not entries:
                return 0.0
            
            if self._current_index >= len(entries):
                return 1.0
            
            return self._current_index / len(entries)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get playback statistics."""
        with self._lock:
            return {
                **self._stats,
                "is_playing": self._is_playing,
                "is_paused": self._is_paused,
                "current_index": self._current_index,
                "total_entries": len(self.recorder.get_entries()),
                "playback_speed": self.playback_speed,
                "progress": self.get_progress()
            }
    
    def get_entry_at_index(self, index: int) -> Optional[Tuple[Any, float, Optional[DroneState]]]:
        """
        Get entry at specific index (for manual playback).
        
        Args:
            index: Entry index
            
        Returns:
            Tuple of (data, timestamp, state) or None
        """
        entries = self.recorder.get_entries()
        if 0 <= index < len(entries):
            return self.recorder.decode_entry(entries[index])
        return None
    
    def get_all_images(self) -> List[Tuple[np.ndarray, float, Optional[DroneState]]]:
        """Get all images from recording with optional synchronized states."""
        entries = self.recorder.get_entries(data_type="image")
        results = []
        for entry in entries:
            result = self.recorder.decode_entry(entry)
            if result:
                results.append(result)
        return results
    
    def get_all_pointclouds(self) -> List[Tuple[np.ndarray, float, Optional[DroneState]]]:
        """Get all point clouds from recording with optional synchronized states."""
        entries = self.recorder.get_entries(data_type="pointcloud")
        results = []
        for entry in entries:
            result = self.recorder.decode_entry(entry)
            if result:
                results.append(result)
        return results
    
    def get_all_states(self) -> List[Tuple[DroneState, float, None]]:
        """Get all states from recording."""
        entries = self.recorder.get_entries(data_type="state")
        results = []
        for entry in entries:
            result = self.recorder.decode_entry(entry)
            if result:
                results.append(result)
        return results

