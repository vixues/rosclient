#!/usr/bin/env python
"""
Modern GUI test tool for RosClient using Pygame + ImGui.

This is an alternative implementation using pyimgui instead of custom Pygame widgets.
It provides a modern, professional UI with all the same functionality as the Pygame version.

Requirements:
    pip install imgui[pygame]
    pip install pygame
    pip install opencv-python  # Optional, for image display
    pip install matplotlib numpy  # Optional, for point cloud display

Usage:
    python rosclient_gui_imgui.py
"""
import pygame
import threading
import json
import time
from typing import Optional, Dict, Any, List, Tuple
import queue
import math

try:
    from rosclient import RosClient, MockRosClient, DroneState, ConnectionState
except ImportError:
    print("Warning: rosclient module not found. Please ensure it's in the Python path.")
    raise

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: cv2 not available. Image display will be disabled.")


import imgui
from imgui.integrations.pygame import PygameRenderer
import OpenGL.GL as gl


try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Point cloud display will be disabled.")


class RosClientImGuiGUI:
    """Modern ImGui GUI for RosClient."""
    
    def __init__(self):
            
        pygame.init()
        self.screen_width = 1400
        self.screen_height = 900
        # Use OpenGL mode for ImGui
        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height),
            pygame.OPENGL | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("ROS Client - ImGui GUI")
        
        # Initialize OpenGL
        gl.glClearColor(0.1, 0.1, 0.1, 1.0)
        
        # Initialize ImGui
        imgui.create_context()
        io = imgui.get_io()
        io.display_size = (self.screen_width, self.screen_height)
        self.impl = PygameRenderer()
        
        # Setup ImGui style (dark theme)
        self.setup_style()
        
        self.clock = pygame.time.Clock()
        self.dt = 1.0 / 60.0  # Initialize delta time
        self.running = True
        
        # Client state
        self.client: Optional[RosClient] = None
        self.is_connected = False
        self.update_thread: Optional[threading.Thread] = None
        self.stop_update = threading.Event()
        self.current_image = None
        
        # UI state
        self.current_tab = 0
        self.tabs = ["Connection", "Status", "Image", "Control", "Point Cloud", "Network Test"]
        
        # Connection tab state
        self.connection_url = "ws://localhost:9090"
        self.use_mock = False
        self.connection_logs = []
        
        # Status tab state
        self.status_values = {}
        
        # Control tab state
        self.topics = []  # List of (name, type) tuples
        self.selected_topic_index = -1
        self.control_topic = "/control"
        self.control_type = "controller_msgs/cmd"
        self.json_message = '{\n    "cmd": 1\n}'
        self.command_history = []
        
        # Point cloud state
        self.current_point_cloud = None
        self.pc_texture = None
        
        # Network test state
        self.test_url = "ws://localhost:9090"
        self.test_timeout = "5"
        self.test_results = []
        
        # Setup update loop
        self.setup_update_loop()
        
    def setup_style(self):
        """Setup ImGui dark theme style."""
        style = imgui.get_style()
        
        # Helper function to safely set color if constant exists
        def set_color_if_exists(color_name, color_value):
            if hasattr(imgui, color_name):
                try:
                    style.colors[getattr(imgui, color_name)] = color_value
                except (KeyError, IndexError):
                    pass  # Color index doesn't exist in this version
        
        # Dark color scheme
        set_color_if_exists('COLOR_WINDOW_BACKGROUND', (0.06, 0.06, 0.08, 1.0))
        set_color_if_exists('COLOR_CHILD_BACKGROUND', (0.12, 0.12, 0.15, 1.0))
        set_color_if_exists('COLOR_POPUP_BACKGROUND', (0.12, 0.12, 0.15, 1.0))
        set_color_if_exists('COLOR_BORDER', (0.20, 0.20, 0.24, 1.0))
        set_color_if_exists('COLOR_FRAME_BACKGROUND', (0.16, 0.16, 0.19, 1.0))
        set_color_if_exists('COLOR_FRAME_BACKGROUND_HOVERED', (0.18, 0.18, 0.22, 1.0))
        set_color_if_exists('COLOR_FRAME_BACKGROUND_ACTIVE', (0.20, 0.20, 0.24, 1.0))
        set_color_if_exists('COLOR_TITLE_BACKGROUND', (0.12, 0.12, 0.15, 1.0))
        set_color_if_exists('COLOR_TITLE_BACKGROUND_ACTIVE', (0.16, 0.16, 0.19, 1.0))
        set_color_if_exists('COLOR_MENUBAR_BACKGROUND', (0.10, 0.10, 0.13, 1.0))
        set_color_if_exists('COLOR_SCROLLBAR_BACKGROUND', (0.10, 0.10, 0.13, 1.0))
        set_color_if_exists('COLOR_SCROLLBAR_GRAB', (0.39, 0.71, 0.96, 1.0))
        set_color_if_exists('COLOR_SCROLLBAR_GRAB_HOVERED', (0.51, 0.83, 1.0, 1.0))
        set_color_if_exists('COLOR_SCROLLBAR_GRAB_ACTIVE', (0.26, 0.65, 0.96, 1.0))
        set_color_if_exists('COLOR_CHECK_MARK', (0.39, 0.71, 0.96, 1.0))
        set_color_if_exists('COLOR_SLIDER_GRAB', (0.39, 0.71, 0.96, 1.0))
        set_color_if_exists('COLOR_SLIDER_GRAB_ACTIVE', (0.26, 0.65, 0.96, 1.0))
        set_color_if_exists('COLOR_BUTTON', (0.39, 0.71, 0.96, 0.40))
        set_color_if_exists('COLOR_BUTTON_HOVERED', (0.39, 0.71, 0.96, 0.60))
        set_color_if_exists('COLOR_BUTTON_ACTIVE', (0.26, 0.65, 0.96, 0.80))
        set_color_if_exists('COLOR_HEADER', (0.39, 0.71, 0.96, 0.31))
        set_color_if_exists('COLOR_HEADER_HOVERED', (0.39, 0.71, 0.96, 0.47))
        set_color_if_exists('COLOR_HEADER_ACTIVE', (0.26, 0.65, 0.96, 0.63))
        set_color_if_exists('COLOR_SEPARATOR', (0.20, 0.20, 0.24, 1.0))
        set_color_if_exists('COLOR_SEPARATOR_HOVERED', (0.39, 0.71, 0.96, 0.78))
        set_color_if_exists('COLOR_SEPARATOR_ACTIVE', (0.26, 0.65, 0.96, 1.0))
        set_color_if_exists('COLOR_RESIZE_GRIP', (0.39, 0.71, 0.96, 0.20))
        set_color_if_exists('COLOR_RESIZE_GRIP_HOVERED', (0.39, 0.71, 0.96, 0.67))
        set_color_if_exists('COLOR_RESIZE_GRIP_ACTIVE', (0.26, 0.65, 0.96, 0.95))
        set_color_if_exists('COLOR_TAB', (0.16, 0.16, 0.19, 1.0))
        set_color_if_exists('COLOR_TAB_HOVERED', (0.39, 0.71, 0.96, 0.80))
        set_color_if_exists('COLOR_TAB_ACTIVE', (0.20, 0.20, 0.24, 1.0))
        set_color_if_exists('COLOR_TAB_UNFOLDED', (0.39, 0.71, 0.96, 0.51))
        set_color_if_exists('COLOR_PLOT_LINES', (0.39, 0.71, 0.96, 1.0))
        set_color_if_exists('COLOR_PLOT_LINES_HOVERED', (0.51, 0.83, 1.0, 1.0))
        set_color_if_exists('COLOR_PLOT_HISTOGRAM', (0.39, 0.71, 0.96, 1.0))
        set_color_if_exists('COLOR_PLOT_HISTOGRAM_HOVERED', (0.51, 0.83, 1.0, 1.0))
        set_color_if_exists('COLOR_TEXT_SELECTED_BACKGROUND', (0.39, 0.71, 0.96, 0.35))
        set_color_if_exists('COLOR_DRAG_DROP_TARGET', (0.39, 0.71, 0.96, 0.95))
        set_color_if_exists('COLOR_NAV_HIGHLIGHT', (0.39, 0.71, 0.96, 1.0))
        set_color_if_exists('COLOR_NAV_WINDOWING_HIGHLIGHT', (0.39, 0.71, 0.96, 0.70))
        set_color_if_exists('COLOR_NAV_WINDOWING_DIM_BG', (0.20, 0.20, 0.20, 0.20))
        set_color_if_exists('COLOR_MODAL_WINDOW_DIM_BG', (0.20, 0.20, 0.20, 0.35))
        
        # Text colors
        set_color_if_exists('COLOR_TEXT', (1.0, 1.0, 1.0, 1.0))
        set_color_if_exists('COLOR_TEXT_DISABLED', (0.50, 0.50, 0.50, 1.0))
        
        # Rounding
        style.window_rounding = 8.0
        style.frame_rounding = 4.0
        style.scrollbar_rounding = 4.0
        style.grab_rounding = 4.0
        style.tab_rounding = 4.0
        
    def setup_update_loop(self):
        """Setup periodic update loop."""
        def update_loop():
            while not self.stop_update.is_set():
                try:
                    if self.is_connected and self.client:
                        if self.current_tab == 1:  # Status tab
                            self.update_status()
                        if self.current_tab == 2 and HAS_CV2:  # Image tab
                            self.update_image()
                        if self.current_tab == 4 and HAS_MATPLOTLIB:  # Point cloud tab
                            self.update_pointcloud()
                        if self.current_tab == 3:  # Control tab - update topic list
                            self.update_topic_list()
                except Exception as e:
                    print(f"Update error: {e}")
                time.sleep(1)
                
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
        
    def connect(self):
        """Connect to ROS bridge."""
        url = self.connection_url.strip()
        if not url:
            self.add_log("Error: Please enter WebSocket address")
            return
            
        def connect_thread():
            try:
                self.add_log(f"Connecting to {url}...")
                
                if self.use_mock:
                    self.client = MockRosClient(url)
                    self.add_log("Using Mock Client (Test Mode)")
                else:
                    self.client = RosClient(url)
                    self.client.connect_async()
                    
                time.sleep(2)
                
                if self.client.is_connected():
                    self.is_connected = True
                    self.add_log("Connection successful!")
                    self.update_topic_list()
                else:
                    self.add_log("Connection failed, please check address and network")
            except Exception as e:
                self.add_log(f"Connection error: {e}")
                
        threading.Thread(target=connect_thread, daemon=True).start()
        
    def disconnect(self):
        """Disconnect from ROS bridge."""
        try:
            if self.client:
                self.client.terminate()
                self.add_log("Disconnected")
            self.is_connected = False
            self.client = None
        except Exception as e:
            self.add_log(f"Disconnect error: {e}")
            
    def add_log(self, message: str):
        """Add log message."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.connection_logs.append(log_entry)
        if len(self.connection_logs) > 50:
            self.connection_logs.pop(0)
            
    def update_status(self):
        """Update status display."""
        if not self.client or not self.is_connected:
            return
            
        try:
            state = self.client.get_status()
            pos = self.client.get_position()
            ori = self.client.get_orientation()
            
            self.status_values = {
                "connected": "Connected" if state.connected else "Disconnected",
                "armed": "Armed" if state.armed else "Disarmed",
                "mode": state.mode or "N/A",
                "battery": f"{state.battery:.1f}%",
                "latitude": f"{pos[0]:.6f}",
                "longitude": f"{pos[1]:.6f}",
                "altitude": f"{pos[2]:.2f}m",
                "roll": f"{ori[0]:.2f}°",
                "pitch": f"{ori[1]:.2f}°",
                "yaw": f"{ori[2]:.2f}°",
                "landed": "Landed" if state.landed else "Flying",
                "reached": "Yes" if state.reached else "No",
                "returned": "Yes" if state.returned else "No",
                "tookoff": "Yes" if state.tookoff else "No",
            }
        except Exception as e:
            pass
            
    def update_image(self):
        """Update image display."""
        if not self.client or not self.is_connected:
            return
            
        try:
            image_data = self.client.get_latest_image()
            if image_data:
                frame, timestamp = image_data
                # Resize image to fit display
                max_width, max_height = 800, 600
                h, w = frame.shape[:2]
                scale = min(max_width / w, max_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                frame_resized = cv2.resize(frame, (new_w, new_h))
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                # Convert numpy array to pygame surface
                frame_rotated = np.rot90(frame_rgb, k=3)
                frame_flipped = np.fliplr(frame_rotated)
                self.current_image = pygame.surfarray.make_surface(frame_flipped)
        except Exception:
            pass
            
    def update_pointcloud(self):
        """Update point cloud display."""
        if not self.client or not self.is_connected:
            return
            
        try:
            pc_data = self.client.get_latest_point_cloud()
            if pc_data:
                points, timestamp = pc_data
                self.current_point_cloud = points
                self.render_pointcloud()
        except Exception:
            pass
            
    def render_pointcloud(self):
        """Render point cloud to pygame surface."""
        if not HAS_MATPLOTLIB or self.current_point_cloud is None:
            return
            
        try:
            # Create matplotlib figure
            fig = Figure(figsize=(8, 6), dpi=100, facecolor='black')
            ax = fig.add_subplot(111, projection='3d')
            
            # Sample points if too many
            points = self.current_point_cloud
            if len(points) > 10000:
                indices = np.random.choice(len(points), 10000, replace=False)
                points = points[indices]
                
            # Plot points
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=points[:, 2], cmap='viridis', s=1)
            ax.set_xlabel('X', color='white')
            ax.set_ylabel('Y', color='white')
            ax.set_zlabel('Z', color='white')
            ax.set_title(f'Point Cloud ({len(points)} points)', color='white')
            ax.set_facecolor('black')
            ax.tick_params(colors='white')
            
            # Convert to pygame surface
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = canvas.buffer_rgba()
            size = canvas.get_width_height()
            self.pc_texture = pygame.image.frombuffer(buf, size, "RGBA")
            fig.clear()
        except Exception as e:
            print(f"Point cloud render error: {e}")
            
    def update_topic_list(self):
        """Update topic list from ROS with name and type."""
        try:
            # Always use DEFAULT_TOPICS from config
            from rosclient.clients.config import DEFAULT_TOPICS
            topics = [(topic.name, topic.type) for topic in DEFAULT_TOPICS.values()]
            
            # Add any additional topics from the client if connected
            if self.client and self.is_connected:
                if hasattr(self.client, '_ts_mgr') and self.client._ts_mgr:
                    if hasattr(self.client._ts_mgr, '_topics'):
                        for topic_name in self.client._ts_mgr._topics.keys():
                            # Check if topic already exists
                            if not any(t[0] == topic_name for t in topics):
                                topics.append((topic_name, ""))  # Unknown type
            
            # Sort by topic name
            topics = sorted(topics, key=lambda x: x[0])
            self.topics = topics
        except Exception as e:
            # Fallback to empty list
            self.topics = []
        
    def send_control_command(self):
        """Send control command."""
        if not self.client or not self.is_connected:
            self.add_log("Warning: Please connect first")
            return
            
        try:
            topic = self.control_topic.strip()
            topic_type = self.control_type.strip()
            message_text = self.json_message.strip()
            
            if not topic or not message_text:
                self.add_log("Warning: Please fill in Topic and message content")
                return
                
            # Parse JSON
            try:
                message = json.loads(message_text)
            except json.JSONDecodeError as e:
                self.add_log(f"Error: JSON format error: {e}")
                return
                
            # Send command
            self.client.publish(topic, topic_type, message)
            
            # Log to history
            timestamp = time.strftime("%H:%M:%S")
            self.command_history.append(
                f"[{timestamp}] {topic} ({topic_type}): {message_text}"
            )
            if len(self.command_history) > 20:
                self.command_history.pop(0)
                
            self.add_log(f"Command sent: {topic} -> {message_text}")
        except Exception as e:
            self.add_log(f"Send command error: {e}")
            
    def test_connection(self):
        """Test ROS connection."""
        url = self.test_url.strip()
        timeout = float(self.test_timeout or "5")
        
        def run_test():
            try:
                self.test_results.append(f"Starting connection test: {url}")
                test_client = RosClient(url)
                test_client.connect_async()
                
                start_time = time.time()
                while time.time() - start_time < timeout:
                    if test_client.is_connected():
                        self.test_results.append("Connection successful!")
                        test_client.terminate()
                        return
                    time.sleep(0.5)
                    
                self.test_results.append("Connection timeout")
                test_client.terminate()
            except Exception as e:
                self.test_results.append(f"Connection failed: {e}")
                
        threading.Thread(target=run_test, daemon=True).start()
        
    def draw_connection_tab(self):
        """Draw connection configuration tab."""
        imgui.text("Connection Settings")
        imgui.separator()
        imgui.spacing()
        
        # WebSocket address input
        imgui.text("WebSocket Address:")
        changed, self.connection_url = imgui.input_text("##url", self.connection_url, 256)
        
        imgui.spacing()
        
        # Mock checkbox
        changed, self.use_mock = imgui.checkbox("Use Mock Client (Test Mode)", self.use_mock)
        
        imgui.spacing()
        
        # Connect/Disconnect buttons
        button_width = 150
        if imgui.button("Connect", button_width, 40):
            self.connect()
        imgui.same_line()
        if imgui.button("Disconnect", button_width, 40):
            self.disconnect()
            
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        
        # Connection log
        imgui.text("Connection Log")
        imgui.separator()
        
        # Log display area with scrolling
        imgui.begin_child("log_area", 0, -1, border=True, flags=imgui.WINDOW_HORIZONTAL_SCROLLING_BAR)
        for log in self.connection_logs[-50:]:  # Show last 50 logs
            imgui.text_wrapped(log)
        imgui.end_child()
        
    def draw_status_tab(self):
        """Draw status monitoring tab."""
        # Connection indicator
        if self.is_connected:
            imgui.text_colored("● Connected", 0.0, 1.0, 0.0, 1.0)
        else:
            imgui.text_colored("● Disconnected", 1.0, 0.0, 0.0, 1.0)
            
        imgui.separator()
        imgui.spacing()
        
        # Status fields in two columns
        status_fields = [
            ("Connection", "connected"),
            ("Armed", "armed"),
            ("Mode", "mode"),
            ("Battery", "battery"),
            ("Latitude", "latitude"),
            ("Longitude", "longitude"),
            ("Altitude", "altitude"),
            ("Roll", "roll"),
            ("Pitch", "pitch"),
            ("Yaw", "yaw"),
            ("Landed", "landed"),
            ("Reached", "reached"),
            ("Returned", "returned"),
            ("Tookoff", "tookoff"),
        ]
        
        # Display in two columns with table for better alignment
        imgui.columns(2, "status_columns", border=True)
        imgui.set_column_width(0, 200)
        
        for label, field in status_fields:
            value = self.status_values.get(field, "N/A")
            
            # Color coding
            if field == "connected":
                color = (0.0, 1.0, 0.0, 1.0) if value == "Connected" else (1.0, 0.0, 0.0, 1.0)
            elif field == "armed":
                color = (1.0, 0.76, 0.03, 1.0) if value == "Armed" else (0.7, 0.7, 0.7, 1.0)
            elif field == "battery":
                try:
                    battery_val = float(value.replace('%', ''))
                    if battery_val < 20:
                        color = (1.0, 0.0, 0.0, 1.0)
                    elif battery_val < 50:
                        color = (1.0, 0.76, 0.03, 1.0)
                    else:
                        color = (0.0, 1.0, 0.0, 1.0)
                except:
                    color = (1.0, 1.0, 1.0, 1.0)
            else:
                color = (1.0, 1.0, 1.0, 1.0)
                
            imgui.text(f"{label}:")
            imgui.next_column()
            imgui.text_colored(str(value), *color)
            imgui.next_column()
            
        imgui.columns(1)
        
    def draw_image_tab(self):
        """Draw image display tab."""
        if not HAS_CV2:
            imgui.text_colored("cv2 not available. Image display is disabled.", 1.0, 0.0, 0.0, 1.0)
            return
            
        imgui.text("Camera Feed")
        imgui.separator()
        
        if self.current_image:
            # Get image dimensions
            img_width, img_height = self.current_image.get_size()
            
            # Get available space
            avail_width = imgui.get_content_region_available_width()
            avail_height = imgui.get_content_region_available_height() - 80
            
            # Calculate display size maintaining aspect ratio
            scale = min(avail_width / img_width, avail_height / img_height, 1.0)
            display_width = int(img_width * scale)
            display_height = int(img_height * scale)
            
            # Convert pygame surface to numpy array for display
            # Note: For full texture support, you would need to use OpenGL textures
            # Here we show image info and use a workaround
            imgui.text(f"Image size: {img_width}x{img_height}")
            imgui.text(f"Display size: {display_width}x{display_height}")
            imgui.spacing()
            
            # Create a child window for image area
            imgui.begin_child("image_area", 0, display_height, border=True)
            
            # Since direct texture rendering requires OpenGL setup,
            # we'll display the image info and use pygame to render it separately
            # In a full implementation, you'd convert the surface to an OpenGL texture
            imgui.text("Image received and processed.")
            imgui.text("(For full image display, OpenGL texture conversion is needed)")
            imgui.text(f"Scale: {scale:.2f}")
            
            imgui.end_child()
            
            # Note: To display the actual image, you would need to:
            # 1. Convert pygame surface to OpenGL texture
            # 2. Use imgui.image() or imgui.image_button() with the texture ID
            # This requires additional OpenGL setup which is beyond basic pygame integration
        else:
            imgui.text("Waiting for image data...")
            if self.is_connected:
                imgui.text_colored("Fetching image...", 0.39, 0.71, 0.96, 1.0)
            else:
                imgui.text_colored("Please connect to ROS bridge first", 0.7, 0.7, 0.7, 1.0)
        
    def draw_control_tab(self):
        """Draw control command tab."""
        # Split into two columns
        imgui.columns(2, "control_columns", border=True)
        imgui.set_column_width(0, 350)
        
        # Left column: Topic list
        imgui.text("Available Topics")
        imgui.separator()
        
        imgui.begin_child("topic_list", 0, -1, border=True)
        for i, (topic_name, topic_type) in enumerate(self.topics):
            is_selected = (i == self.selected_topic_index)
            display_text = topic_name
            if topic_type:
                display_text += f" ({topic_type})"
            if imgui.selectable(f"{display_text}##{i}", is_selected)[0]:
                self.selected_topic_index = i
                self.control_topic = topic_name
                if topic_type:
                    self.control_type = topic_type
        imgui.end_child()
        
        imgui.next_column()
        
        # Right column: Configuration and editor
        imgui.text("Topic Configuration")
        imgui.separator()
        
        imgui.text("Topic Name:")
        changed, self.control_topic = imgui.input_text("##topic", self.control_topic, 256)
        
        imgui.spacing()
        
        imgui.text("Topic Type:")
        changed, self.control_type = imgui.input_text("##type", self.control_type, 256)
        
        imgui.separator()
        imgui.spacing()
        
        imgui.text("Message Content (JSON):")
        imgui.separator()
        
        # JSON editor (multiline text input)
        available_height = imgui.get_content_region_available_height() - 250
        changed, self.json_message = imgui.input_text_multiline(
            "##json", self.json_message, 10000, 
            height=max(200, available_height),
            flags=imgui.INPUT_TEXT_ALLOW_TAB_INPUT
        )
        
        imgui.spacing()
        
        # Preset buttons
        button_width = 100
        if imgui.button("Takeoff", button_width, 30):
            self.json_message = '{\n    "cmd": 1\n}'
        imgui.same_line()
        if imgui.button("Land", button_width, 30):
            self.json_message = '{\n    "cmd": 2\n}'
        imgui.same_line()
        if imgui.button("Return", button_width, 30):
            self.json_message = '{\n    "cmd": 3\n}'
        imgui.same_line()
        if imgui.button("Hover", button_width, 30):
            self.json_message = '{\n    "cmd": 4\n}'
            
        imgui.spacing()
        
        # Format and Send buttons
        if imgui.button("Format JSON", 130, 30):
            try:
                obj = json.loads(self.json_message)
                self.json_message = json.dumps(obj, indent=4)
            except Exception as e:
                self.add_log(f"Error: Invalid JSON format: {e}")
                
        imgui.same_line()
        if imgui.button("Send Command", 150, 30):
            self.send_control_command()
            
        imgui.columns(1)
        imgui.separator()
        imgui.spacing()
        
        # Command history
        imgui.text("Command History")
        imgui.separator()
        
        imgui.begin_child("history", 0, 150, border=True, flags=imgui.WINDOW_HORIZONTAL_SCROLLING_BAR)
        for cmd in self.command_history[-15:]:  # Show last 15 commands
            imgui.text_wrapped(cmd)
        imgui.end_child()
        
    def draw_pointcloud_tab(self):
        """Draw point cloud display tab."""
        if not HAS_MATPLOTLIB:
            imgui.text_colored("matplotlib and numpy required for point cloud display", 
                            1.0, 0.0, 0.0, 1.0)
            return
            
        imgui.text("Point Cloud Visualization")
        imgui.separator()
        imgui.spacing()
            
        if self.pc_texture:
            # Similar to image display, would need texture conversion for full display
            point_count = len(self.current_point_cloud) if self.current_point_cloud is not None else 0
            imgui.text(f"Point cloud rendered: {point_count} points")
            imgui.text("(For full visualization, OpenGL texture conversion is needed)")
        else:
            imgui.text("Waiting for point cloud data...")
            if self.is_connected:
                imgui.text_colored("Fetching point cloud...", 0.39, 0.71, 0.96, 1.0)
            else:
                imgui.text_colored("Please connect to ROS bridge first", 0.7, 0.7, 0.7, 1.0)
        
    def draw_network_tab(self):
        """Draw network test tab."""
        imgui.text("Test Configuration")
        imgui.separator()
        imgui.spacing()
        
        imgui.text("Test Address:")
        changed, self.test_url = imgui.input_text("##test_url", self.test_url, 256)
        
        imgui.spacing()
        
        imgui.text("Timeout (seconds):")
        changed, self.test_timeout = imgui.input_text("##timeout", self.test_timeout, 32)
        
        imgui.spacing()
        
        if imgui.button("Test Connection", 150, 35):
            self.test_connection()
            
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        
        # Test results
        imgui.text("Test Results")
        imgui.separator()
        
        imgui.begin_child("test_results", 0, -1, border=True, flags=imgui.WINDOW_HORIZONTAL_SCROLLING_BAR)
        for result in self.test_results[-25:]:  # Show last 25 results
            imgui.text_wrapped(result)
        imgui.end_child()
        
    def draw(self):
        """Draw the main window with tabs."""
        # Clear OpenGL buffer
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        
        # Update ImGui IO with delta time
        io = imgui.get_io()
        io.delta_time = self.dt
        
        imgui.new_frame()
        
        # Main window - fullscreen
        imgui.set_next_window_size(self.screen_width, self.screen_height)
        imgui.set_next_window_position(0, 0)
        
        imgui.begin("ROS Client", 
                   flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | 
                         imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE | 
                         imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS)
        
        # Tab bar using tabs API for better appearance
        if imgui.begin_tab_bar("MainTabs"):
            for i, tab_name in enumerate(self.tabs):
                tab_flags = 0
                if i == self.current_tab:
                    tab_flags = imgui.TAB_ITEM_SET_SELECTED
                    
                opened, selected = imgui.begin_tab_item(tab_name, flags=tab_flags)
                if opened:
                    self.current_tab = i
                    imgui.end_tab_item()
                    
            imgui.end_tab_bar()
        
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        
        # Draw current tab content in a child window for better scrolling
        imgui.begin_child("tab_content", 0, -1, border=False)
        
        if self.current_tab == 0:
            self.draw_connection_tab()
        elif self.current_tab == 1:
            self.draw_status_tab()
        elif self.current_tab == 2:
            self.draw_image_tab()
        elif self.current_tab == 3:
            self.draw_control_tab()
        elif self.current_tab == 4:
            self.draw_pointcloud_tab()
        elif self.current_tab == 5:
            self.draw_network_tab()
            
        imgui.end_child()
        imgui.end()
        
        # Render
        imgui.render()
        self.impl.render(imgui.get_draw_data())
        
        pygame.display.flip()
        
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            self.impl.process_event(event)
            
    def update(self):
        """Update game state."""
        self.dt = self.clock.tick(60) / 1000.0  # Convert to seconds
        
    def run(self):
        """Main game loop."""
        if not self.connection_logs:
            self.add_log("Waiting for connection...")
            
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            
        # Cleanup
        self.stop_update.set()
        if self.client:
            self.disconnect()
        self.impl.shutdown()
        pygame.quit()


def main():
    """Main entry point."""
    app = RosClientImGuiGUI()
    app.run()


if __name__ == "__main__":
    main()

