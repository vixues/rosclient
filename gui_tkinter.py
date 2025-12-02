#!/usr/bin/env python
"""GUI test tool for RosClient."""
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import json
import time
from typing import Optional, Dict, Any, Tuple
import queue

try:
    from rosclient import RosClient, MockRosClient, DroneState, ConnectionState
    try:
        from rosclient.clients import AirSimClient
        HAS_AIRSIM = True
    except ImportError:
        HAS_AIRSIM = False
        AirSimClient = None
except ImportError:
    print("Warning: rosclient module not found. Please ensure it's in the Python path.")
    raise

try:
    import cv2
    from PIL import Image, ImageTk
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: cv2 or PIL not available. Image display will be disabled.")

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Point cloud display will be disabled.")


class RosClientGUITest:
    """GUI test tool for RosClient."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("ROS Client GUI - Drone Control Console")
        self.root.geometry("1500x950")
        self.root.minsize(1200, 700)
        
        # 配置样式
        self.setup_styles()
        
        self.client: Optional[RosClient] = None
        self.is_connected = False
        self.update_thread: Optional[threading.Thread] = None
        self.image_update_thread: Optional[threading.Thread] = None
        self.stop_update = threading.Event()
        self.image_queue = queue.Queue(maxsize=1)
        self.point_cloud_queue = queue.Queue(maxsize=1)
        
        # Image display optimization: track canvas size to avoid unnecessary resizing
        self._last_canvas_size: Dict[str, Tuple[int, int]] = {}  # Track canvas size changes
        self._pending_image_update: Dict[str, bool] = {"main": False, "control": False}  # Separate flags for each canvas
        
        self.setup_ui()
        self.setup_update_loop()
        
    def setup_styles(self):
        """Setup modern style theme"""
        style = ttk.Style()
        
        # Try to use modern theme
        available_themes = style.theme_names()
        if 'vista' in available_themes:
            style.theme_use('vista')
        elif 'clam' in available_themes:
            style.theme_use('clam')
        
        # Configure color scheme
        style.configure('Title.TLabel')
        style.configure('Heading.TLabel')
        style.configure('Status.TLabel')
        style.configure('Success.TLabel', foreground='#28a745')
        style.configure('Error.TLabel', foreground='#dc3545')
        style.configure('Warning.TLabel', foreground='#ffc107')
        style.configure('Info.TLabel', foreground='#17a2b8')
        
        # Configure button styles
        style.configure('Primary.TButton')
        style.configure('Success.TButton', foreground='white')
        style.configure('Danger.TButton', foreground='white')
        
        # Configure Notebook style
        style.configure('TNotebook.Tab', padding=[12, 6])
        style.map('TNotebook.Tab', 
                  expand=[('selected', [1, 1, 1, 0])])
        
    def setup_ui(self):
        """Setup the user interface."""
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Create status bar
        self.setup_statusbar(main_container)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_container)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Connection tab
        self.connection_frame = ttk.Frame(notebook)
        notebook.add(self.connection_frame, text="Connection")
        self.setup_connection_tab()
        
        # Status tab
        self.status_frame = ttk.Frame(notebook)
        notebook.add(self.status_frame, text="Status")
        self.setup_status_tab()
        
        # Image tab
        self.image_frame = ttk.Frame(notebook)
        notebook.add(self.image_frame, text="Image")
        self.setup_image_tab()
        
        # Point Cloud tab
        self.pointcloud_frame = ttk.Frame(notebook)
        notebook.add(self.pointcloud_frame, text="Point Cloud")
        self.setup_pointcloud_tab()
        
        # Control tab
        self.control_frame = ttk.Frame(notebook)
        notebook.add(self.control_frame, text="Control")
        self.setup_control_tab()
        
        # Network Test tab
        self.network_frame = ttk.Frame(notebook)
        notebook.add(self.network_frame, text="Network Test")
        self.setup_network_tab()
        
        # Recording tab
        self.recording_frame = ttk.Frame(notebook)
        notebook.add(self.recording_frame, text="Recording")
        self.setup_recording_tab()
        
    def setup_statusbar(self, parent):
        """Create status bar"""
        statusbar = ttk.Frame(parent)
        statusbar.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))
        
        # Connection status indicator
        status_frame = ttk.Frame(statusbar)
        status_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT, padx=(0, 5))
        
        # Status indicator dot
        self.status_indicator = tk.Canvas(status_frame, width=16, height=16, highlightthickness=0)
        self.status_indicator.pack(side=tk.LEFT, padx=5)
        self.status_indicator.create_oval(4, 4, 12, 12, fill='#dc3545', outline='')
        self.status_indicator_text = ttk.Label(status_frame, text="Disconnected")
        self.status_indicator_text.pack(side=tk.LEFT, padx=5)
        
        # Separator
        ttk.Separator(statusbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Time display
        self.time_label = ttk.Label(statusbar, text="")
        self.time_label.pack(side=tk.LEFT, padx=10)
        
        # Right side info
        info_frame = ttk.Frame(statusbar)
        info_frame.pack(side=tk.RIGHT, padx=10, pady=5)
        
        self.client_type_label = ttk.Label(info_frame, text="Client: None", foreground="#6c757d")
        self.client_type_label.pack(side=tk.RIGHT, padx=(0, 10))
        
        self.info_label = ttk.Label(info_frame, text="Ready")
        self.info_label.pack(side=tk.RIGHT)
        
        # Update time display
        self.update_time_display()
        
    def update_time_display(self):
        """Update status bar time display"""
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_time_display)
        
    def setup_connection_tab(self):
        """Setup connection configuration tab."""
        # Main container
        main_pane = ttk.PanedWindow(self.connection_frame, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side: connection settings
        left_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=1)
        
        # Connection settings
        conn_group = ttk.LabelFrame(left_frame, text="Connection Settings", padding=15)
        conn_group.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Connection address
        url_frame = ttk.Frame(conn_group)
        url_frame.pack(fill=tk.X, pady=8)
        self.connection_label = ttk.Label(url_frame, text="WebSocket URL:")
        self.connection_label.pack(side=tk.LEFT, padx=(0, 10))
        self.connection_url = tk.StringVar(value="ws://192.168.27.152:9090")
        url_entry = ttk.Entry(url_frame, textvariable=self.connection_url, width=45)
        url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Connection mode
        mode_frame = ttk.Frame(conn_group)
        mode_frame.pack(fill=tk.X, pady=8)
        
        # Client type selection
        client_type_frame = ttk.LabelFrame(mode_frame, text="Client Type", padding=5)
        client_type_frame.pack(fill=tk.X, pady=5)
        
        self.client_type = tk.StringVar(value="ros")
        ttk.Radiobutton(client_type_frame, text="ROS Client", variable=self.client_type, 
                       value="ros", command=self.on_client_type_changed).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(client_type_frame, text="Mock Client (Test)", variable=self.client_type, 
                       value="mock", command=self.on_client_type_changed).pack(side=tk.LEFT, padx=10)
        if HAS_AIRSIM:
            ttk.Radiobutton(client_type_frame, text="AirSim Client", variable=self.client_type, 
                           value="airsim", command=self.on_client_type_changed).pack(side=tk.LEFT, padx=10)
        else:
            ttk.Radiobutton(client_type_frame, text="AirSim Client (Not Available)", 
                           variable=self.client_type, value="airsim", 
                           state=tk.DISABLED).pack(side=tk.LEFT, padx=10)
        
        # Legacy mock checkbox (hidden, kept for compatibility)
        self.use_mock = tk.BooleanVar(value=False)
        
        # Recording file for Mock Client
        self.recording_file_frame = ttk.Frame(conn_group)
        self.recording_file_frame.pack(fill=tk.X, pady=8)
        ttk.Label(self.recording_file_frame, text="Recording File:").pack(side=tk.LEFT, padx=(0, 10))
        self.recording_file_path = tk.StringVar()
        recording_entry = ttk.Entry(self.recording_file_frame, textvariable=self.recording_file_path, 
                                    width=35, state=tk.DISABLED)
        recording_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(self.recording_file_frame, text="Browse...", 
                  command=self.browse_recording_file, width=12).pack(side=tk.LEFT, padx=5)
        self.recording_file_frame.pack_forget()  # Hide by default
        
        # Connection buttons
        btn_frame = ttk.Frame(conn_group)
        btn_frame.pack(fill=tk.X, pady=15)
        
        self.connect_btn = ttk.Button(btn_frame, text="Connect", command=self.connect,
                                     style='Primary.TButton', width=15)
        self.connect_btn.pack(side=tk.LEFT, padx=5)
        
        self.disconnect_btn = ttk.Button(btn_frame, text="Disconnect", command=self.disconnect, 
                                        state=tk.DISABLED, width=15)
        self.disconnect_btn.pack(side=tk.LEFT, padx=5)
        
        # Quick connection buttons
        quick_frame = ttk.LabelFrame(conn_group, text="Quick Connect", padding=10)
        quick_frame.pack(fill=tk.X, pady=10)
        
        # ROS quick connects
        self.ros_quick_urls = [
            ("Local Test", "ws://localhost:9090"),
            ("Default Address", "ws://192.168.27.152:9090"),
        ]
        
        # AirSim quick connects
        self.airsim_quick_urls = [
            ("Local AirSim", "127.0.0.1:41451"),
            ("Default AirSim", "127.0.0.1"),
        ]
        
        self.quick_buttons_frame = ttk.Frame(quick_frame)
        self.quick_buttons_frame.pack(fill=tk.X)
        self.update_quick_connect_buttons()
        
        # Right side: connection log
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=1)
        
        status_group = ttk.LabelFrame(right_frame, text="Connection Log", padding=10)
        status_group.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Log toolbar
        log_toolbar = ttk.Frame(status_group)
        log_toolbar.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(log_toolbar, text="Clear", command=self.clear_log, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(log_toolbar, text="Save", command=self.save_log, width=10).pack(side=tk.LEFT, padx=2)
        
        self.connection_status = scrolledtext.ScrolledText(
            status_group, height=20, width=50, wrap=tk.WORD,
            bg='#f8f9fa', fg='#212529'
        )
        self.connection_status.pack(fill=tk.BOTH, expand=True)
        self.log("Waiting for connection...", "info")
        
    def setup_status_tab(self):
        """Setup status monitoring tab."""
        # Main container
        main_container = ttk.Frame(self.status_frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status display area
        status_group = ttk.LabelFrame(main_container, text="Drone Status", padding=15)
        status_group.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create status labels
        self.status_labels = {}
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
        
        # Use grid layout, 3 columns display
        for i, (label, field) in enumerate(status_fields):
            row = i // 3
            col = (i % 3) * 2
            
            # Label
            label_frame = ttk.Frame(status_group)
            label_frame.grid(row=row, column=col, sticky=tk.W+tk.E, padx=10, pady=8)
            
            ttk.Label(label_frame, text=f"{label}:").pack(side=tk.LEFT)
            
            # Value label
            value_label = ttk.Label(label_frame, text="N/A", 
                                   foreground="#6c757d")
            value_label.pack(side=tk.LEFT, padx=(10, 0))
            self.status_labels[field] = value_label
            
            # Configure column weight
            status_group.columnconfigure(col, weight=1)
            status_group.columnconfigure(col+1, weight=1)
            
        # Control buttons
        btn_frame = ttk.Frame(status_group)
        btn_frame.grid(row=(len(status_fields)//3 + 1), column=0, columnspan=6, pady=15)
        
        ttk.Button(btn_frame, text="Refresh Status", command=self.update_status_display,
                  style='Primary.TButton', width=15).pack(side=tk.LEFT, padx=5)
        
        self.auto_refresh = tk.BooleanVar(value=True)
        ttk.Checkbutton(btn_frame, text="Auto Refresh (1s)", 
                       variable=self.auto_refresh).pack(side=tk.LEFT, padx=10)
        
    def setup_image_tab(self):
        """Setup image display tab."""
        if not HAS_CV2:
            error_frame = ttk.Frame(self.image_frame)
            error_frame.pack(fill=tk.BOTH, expand=True)
            ttk.Label(error_frame, text="Please install opencv-python and Pillow to display images",
                     foreground="red").pack(expand=True)
            return
            
        # Image display
        img_group = ttk.LabelFrame(self.image_frame, text="Camera Image", padding=10)
        img_group.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Image canvas with border
        canvas_frame = ttk.Frame(img_group)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.image_canvas = tk.Canvas(canvas_frame, bg="#1a1a1a", width=640, height=480,
                                     highlightthickness=2, highlightbackground="#dee2e6")
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Placeholder text
        self.image_canvas.create_text(320, 240, text="Waiting for image data...", 
                                      fill="#6c757d")
        
        # Image controls
        img_controls = ttk.Frame(img_group)
        img_controls.pack(fill=tk.X, pady=5)
        
        # Left controls
        left_controls = ttk.Frame(img_controls)
        left_controls.pack(side=tk.LEFT)
        
        self.auto_update_image = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_controls, text="Auto Update", 
                       variable=self.auto_update_image).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(left_controls, text="Fetch Manual", 
                  command=self.fetch_image_manual,
                  width=12).pack(side=tk.LEFT, padx=5)
        
        # Right info
        right_controls = ttk.Frame(img_controls)
        right_controls.pack(side=tk.RIGHT)
        
        self.image_info_label = ttk.Label(right_controls, text="Waiting for image...",
                                          foreground="#6c757d")
        self.image_info_label.pack(side=tk.RIGHT, padx=10)
        
    def setup_pointcloud_tab(self):
        """Setup point cloud display tab."""
        if not HAS_MATPLOTLIB:
            error_frame = ttk.Frame(self.pointcloud_frame)
            error_frame.pack(fill=tk.BOTH, expand=True)
            ttk.Label(error_frame, text="Please install matplotlib and numpy to display point cloud",
                     foreground="red").pack(expand=True)
            return
            
        # Point cloud display
        pc_group = ttk.LabelFrame(self.pointcloud_frame, text="Point Cloud Data", padding=10)
        pc_group.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Matplotlib figure with dark theme
        self.pc_figure = Figure(figsize=(8, 6), dpi=100, facecolor='#f8f9fa')
        self.pc_ax = self.pc_figure.add_subplot(111, projection='3d', facecolor='#ffffff')
        self.pc_ax.set_xlabel('X (m)', fontsize=9)
        self.pc_ax.set_ylabel('Y (m)', fontsize=9)
        self.pc_ax.set_zlabel('Z (m)', fontsize=9)
        self.pc_ax.set_title('Point Cloud Visualization', fontsize=11, pad=10)
        
        self.pc_canvas = FigureCanvasTkAgg(self.pc_figure, pc_group)
        self.pc_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Point cloud controls
        pc_controls = ttk.Frame(pc_group)
        pc_controls.pack(fill=tk.X, pady=5)
        
        # Left controls
        left_controls = ttk.Frame(pc_controls)
        left_controls.pack(side=tk.LEFT)
        
        self.auto_update_pc = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_controls, text="Auto Update", 
                       variable=self.auto_update_pc).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(left_controls, text="Fetch Manual", 
                  command=self.fetch_pointcloud_manual,
                  width=12).pack(side=tk.LEFT, padx=5)
        
        # Right info
        right_controls = ttk.Frame(pc_controls)
        right_controls.pack(side=tk.RIGHT)
        
        self.pc_info_label = ttk.Label(right_controls, text="Waiting for point cloud data...",
                                       foreground="#6c757d")
        self.pc_info_label.pack(side=tk.RIGHT, padx=10)
        
    def setup_control_tab(self):
        """Setup control command tab."""
        # Main container
        main_pane = ttk.PanedWindow(self.control_frame, orient=tk.VERTICAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Topic configuration
        topic_group = ttk.LabelFrame(main_pane, text="Topic Configuration", padding=12)
        main_pane.add(topic_group, weight=0)
        
        topic_inner = ttk.Frame(topic_group)
        topic_inner.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(topic_inner, text="Topic Name:").grid(
            row=0, column=0, sticky=tk.W, pady=5, padx=5)
        self.control_topic = tk.StringVar(value="/control")
        ttk.Entry(topic_inner, textvariable=self.control_topic, width=35).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        ttk.Label(topic_inner, text="Topic Type:").grid(
            row=0, column=2, sticky=tk.W, padx=(20, 5), pady=5)
        self.control_type = tk.StringVar(value="controller_msgs/cmd")
        ttk.Entry(topic_inner, textvariable=self.control_type, width=35).grid(row=0, column=3, padx=5, pady=5, sticky=tk.W+tk.E)
        
        topic_inner.columnconfigure(1, weight=1)
        topic_inner.columnconfigure(3, weight=1)
        
        # Preset commands (ROS/Mock)
        self.preset_group = ttk.LabelFrame(main_pane, text="Preset Commands (ROS/Mock)", padding=10)
        main_pane.add(self.preset_group, weight=0)
        
        presets = [
            ("Takeoff", '{"cmd": 1}'),
            ("Land", '{"cmd": 2}'),
            ("Return", '{"cmd": 3}'),
            ("Hover", '{"cmd": 4}'),
        ]
        
        for i, (name, cmd) in enumerate(presets):
            btn = ttk.Button(self.preset_group, text=name, 
                           command=lambda c=cmd: self.set_preset_command(c),
                           width=12)
            btn.grid(row=0, column=i, padx=5, pady=5)
        
        # AirSim Quick Actions with image display
        self.airsim_actions_group = ttk.LabelFrame(main_pane, text="AirSim Quick Actions", padding=10)
        # Store reference to main_pane for later use
        self.control_main_pane = main_pane
        # Don't add to pane initially - will be added dynamically
        
        # Create horizontal paned window for controls and image
        airsim_horizontal_pane = ttk.PanedWindow(self.airsim_actions_group, orient=tk.HORIZONTAL)
        airsim_horizontal_pane.pack(fill=tk.BOTH, expand=True)
        
        # Left side: Control buttons
        controls_frame = ttk.Frame(airsim_horizontal_pane)
        airsim_horizontal_pane.add(controls_frame, weight=1)
        
        # Basic control buttons
        basic_frame = ttk.Frame(controls_frame)
        basic_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(basic_frame, text="Basic Control:", font=('', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        ttk.Button(basic_frame, text="Arm", command=self.airsim_arm, width=10).pack(side=tk.LEFT, padx=3)
        ttk.Button(basic_frame, text="Disarm", command=self.airsim_disarm, width=10).pack(side=tk.LEFT, padx=3)
        ttk.Button(basic_frame, text="Takeoff", command=self.airsim_takeoff, width=10, style='Success.TButton').pack(side=tk.LEFT, padx=3)
        ttk.Button(basic_frame, text="Land", command=self.airsim_land, width=10, style='Danger.TButton').pack(side=tk.LEFT, padx=3)
        
        # Position control
        pos_frame = ttk.LabelFrame(controls_frame, text="Position Control", padding=8)
        pos_frame.pack(fill=tk.X, pady=5)
        
        pos_inner = ttk.Frame(pos_frame)
        pos_inner.pack(fill=tk.X)
        
        ttk.Label(pos_inner, text="X:").grid(row=0, column=0, padx=3, pady=3, sticky=tk.W)
        self.airsim_x = tk.StringVar(value="0.0")
        ttk.Entry(pos_inner, textvariable=self.airsim_x, width=10).grid(row=0, column=1, padx=3, pady=3)
        
        ttk.Label(pos_inner, text="Y:").grid(row=0, column=2, padx=3, pady=3, sticky=tk.W)
        self.airsim_y = tk.StringVar(value="0.0")
        ttk.Entry(pos_inner, textvariable=self.airsim_y, width=10).grid(row=0, column=3, padx=3, pady=3)
        
        ttk.Label(pos_inner, text="Z:").grid(row=0, column=4, padx=3, pady=3, sticky=tk.W)
        self.airsim_z = tk.StringVar(value="-5.0")
        ttk.Entry(pos_inner, textvariable=self.airsim_z, width=10).grid(row=0, column=5, padx=3, pady=3)
        
        ttk.Label(pos_inner, text="Velocity:").grid(row=1, column=0, padx=3, pady=3, sticky=tk.W)
        self.airsim_velocity = tk.StringVar(value="5.0")
        ttk.Entry(pos_inner, textvariable=self.airsim_velocity, width=10).grid(row=1, column=1, padx=3, pady=3)
        
        ttk.Button(pos_inner, text="Move To", command=self.airsim_move_to, width=12).grid(row=1, column=2, columnspan=4, padx=5, pady=3, sticky=tk.E)
        
        # Velocity control
        vel_frame = ttk.LabelFrame(controls_frame, text="Velocity Control", padding=8)
        vel_frame.pack(fill=tk.X, pady=5)
        
        vel_inner = ttk.Frame(vel_frame)
        vel_inner.pack(fill=tk.X)
        
        ttk.Label(vel_inner, text="Vx:").grid(row=0, column=0, padx=3, pady=3, sticky=tk.W)
        self.airsim_vx = tk.StringVar(value="0.0")
        ttk.Entry(vel_inner, textvariable=self.airsim_vx, width=10).grid(row=0, column=1, padx=3, pady=3)
        
        ttk.Label(vel_inner, text="Vy:").grid(row=0, column=2, padx=3, pady=3, sticky=tk.W)
        self.airsim_vy = tk.StringVar(value="0.0")
        ttk.Entry(vel_inner, textvariable=self.airsim_vy, width=10).grid(row=0, column=3, padx=3, pady=3)
        
        ttk.Label(vel_inner, text="Vz:").grid(row=0, column=4, padx=3, pady=3, sticky=tk.W)
        self.airsim_vz = tk.StringVar(value="0.0")
        ttk.Entry(vel_inner, textvariable=self.airsim_vz, width=10).grid(row=0, column=5, padx=3, pady=3)
        
        ttk.Label(vel_inner, text="Duration:").grid(row=1, column=0, padx=3, pady=3, sticky=tk.W)
        self.airsim_duration = tk.StringVar(value="1.0")
        ttk.Entry(vel_inner, textvariable=self.airsim_duration, width=10).grid(row=1, column=1, padx=3, pady=3)
        
        ttk.Button(vel_inner, text="Set Velocity", command=self.airsim_set_velocity, width=12).grid(row=1, column=2, columnspan=4, padx=5, pady=3, sticky=tk.E)
        
        # Other actions
        other_frame = ttk.Frame(controls_frame)
        other_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(other_frame, text="Other Actions:", font=('', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        ttk.Button(other_frame, text="Take Photo", command=self.airsim_take_photo, width=12).pack(side=tk.LEFT, padx=3)
        ttk.Button(other_frame, text="Get Telemetry", command=self.airsim_get_telemetry, width=12).pack(side=tk.LEFT, padx=3)
        
        # Right side: Real-time image display (only for AirSim)
        if HAS_CV2:
            image_display_frame = ttk.LabelFrame(airsim_horizontal_pane, text="Real-time Camera View", padding=5)
            airsim_horizontal_pane.add(image_display_frame, weight=1)
            
            # Image canvas
            self.control_image_canvas = tk.Canvas(image_display_frame, bg="#1a1a1a", width=480, height=360,
                                                 highlightthickness=2, highlightbackground="#dee2e6")
            self.control_image_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Placeholder text
            self.control_image_canvas.create_text(240, 180, text="Waiting for image data...", 
                                                 fill="#6c757d", font=('', 10))
            
            # Image info label
            self.control_image_info = ttk.Label(image_display_frame, text="Waiting for image...",
                                               foreground="#6c757d", font=('', 8))
            self.control_image_info.pack(pady=(0, 5))
        else:
            # If CV2 is not available, just add an empty frame
            image_display_frame = ttk.Frame(airsim_horizontal_pane)
            airsim_horizontal_pane.add(image_display_frame, weight=0)
            ttk.Label(image_display_frame, text="OpenCV not available", 
                     foreground="red").pack(expand=True)
        
        # Message editor
        msg_group = ttk.LabelFrame(main_pane, text="Message Content (JSON Format)", padding=10)
        main_pane.add(msg_group, weight=2)
        
        # Editor toolbar
        editor_toolbar = ttk.Frame(msg_group)
        editor_toolbar.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(editor_toolbar, text="Format", command=self.format_json, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(editor_toolbar, text="Clear", command=self.clear_editor, width=10).pack(side=tk.LEFT, padx=2)
        
        self.message_editor = scrolledtext.ScrolledText(
            msg_group, height=12, width=80, wrap=tk.NONE,
            bg='#ffffff', fg='#212529'
        )
        self.message_editor.pack(fill=tk.BOTH, expand=True)
        self.message_editor.insert("1.0", '{\n    "cmd": 1\n}')
        
        # Send button
        btn_frame = ttk.Frame(main_pane)
        main_pane.add(btn_frame, weight=0)
        
        ttk.Button(btn_frame, text="Send Command", command=self.send_control_command,
                  style='Primary.TButton', width=20).pack(pady=10)
        
        # Command history
        history_group = ttk.LabelFrame(main_pane, text="Command History", padding=10)
        main_pane.add(history_group, weight=1)
        
        # History toolbar
        history_toolbar = ttk.Frame(history_group)
        history_toolbar.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(history_toolbar, text="Clear History", command=self.clear_history, width=12).pack(side=tk.LEFT, padx=2)
        
        self.command_history = scrolledtext.ScrolledText(
            history_group, height=8, width=80, wrap=tk.WORD,
            bg='#f8f9fa', fg='#212529'
        )
        self.command_history.pack(fill=tk.BOTH, expand=True)
        
    def setup_network_tab(self):
        """Setup network test tab."""
        # Main container
        main_pane = ttk.PanedWindow(self.network_frame, orient=tk.VERTICAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Test configuration
        test_group = ttk.LabelFrame(main_pane, text="Network Test Configuration", padding=12)
        main_pane.add(test_group, weight=0)
        
        config_inner = ttk.Frame(test_group)
        config_inner.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(config_inner, text="Test URL:").grid(
            row=0, column=0, sticky=tk.W, pady=8, padx=5)
        self.test_url = tk.StringVar(value="ws://192.168.27.152:9090")
        ttk.Entry(config_inner, textvariable=self.test_url, width=45).grid(row=0, column=1, padx=5, pady=8, sticky=tk.W+tk.E)
        
        ttk.Label(config_inner, text="Timeout (s):").grid(
            row=1, column=0, sticky=tk.W, pady=8, padx=5)
        self.test_timeout = tk.StringVar(value="5")
        ttk.Entry(config_inner, textvariable=self.test_timeout, width=15).grid(row=1, column=1, sticky=tk.W, padx=5, pady=8)
        
        config_inner.columnconfigure(1, weight=1)
        
        # Test buttons
        btn_frame = ttk.Frame(test_group)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Test Connection", command=self.test_connection,
                  width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Test Topics", command=self.test_topics,
                  width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Full Test", command=self.run_full_test,
                  style='Primary.TButton', width=15).pack(side=tk.LEFT, padx=5)
        
        # Test results
        result_group = ttk.LabelFrame(main_pane, text="Test Results", padding=10)
        main_pane.add(result_group, weight=1)
        
        # Results toolbar
        result_toolbar = ttk.Frame(result_group)
        result_toolbar.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(result_toolbar, text="Clear", command=self.clear_test_results, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(result_toolbar, text="Save", command=self.save_test_results, width=10).pack(side=tk.LEFT, padx=2)
        
        self.test_results = scrolledtext.ScrolledText(
            result_group, height=20, width=80, wrap=tk.WORD,
            bg='#f8f9fa', fg='#212529'
        )
        self.test_results.pack(fill=tk.BOTH, expand=True)
    
    def setup_recording_tab(self):
        """Setup recording control tab."""
        # Main container
        main_pane = ttk.PanedWindow(self.recording_frame, orient=tk.VERTICAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Recording control
        control_group = ttk.LabelFrame(main_pane, text="Recording Control", padding=15)
        main_pane.add(control_group, weight=0)
        
        # Start/Stop recording
        record_btn_frame = ttk.Frame(control_group)
        record_btn_frame.pack(fill=tk.X, pady=10)
        
        self.start_record_btn = ttk.Button(record_btn_frame, text="Start Recording", 
                                          command=self.start_recording,
                                          style='Success.TButton', width=15)
        self.start_record_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_record_btn = ttk.Button(record_btn_frame, text="Stop Recording", 
                                         command=self.stop_recording,
                                         state=tk.DISABLED, width=15)
        self.stop_record_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_record_btn = ttk.Button(record_btn_frame, text="Save Recording", 
                                         command=self.save_recording,
                                         state=tk.DISABLED, width=15)
        self.save_record_btn.pack(side=tk.LEFT, padx=5)
        
        # Recording options
        options_frame = ttk.Frame(control_group)
        options_frame.pack(fill=tk.X, pady=10)
        
        self.record_images = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Record Images", 
                       variable=self.record_images).pack(side=tk.LEFT, padx=10)
        
        self.record_pointclouds = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Record Point Clouds", 
                       variable=self.record_pointclouds).pack(side=tk.LEFT, padx=10)
        
        self.record_states = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Record States", 
                       variable=self.record_states).pack(side=tk.LEFT, padx=10)
        
        # Recording statistics
        stats_group = ttk.LabelFrame(main_pane, text="Recording Statistics", padding=10)
        main_pane.add(stats_group, weight=1)
        
        self.recording_stats = scrolledtext.ScrolledText(
            stats_group, height=15, width=80, wrap=tk.WORD,
            bg='#f8f9fa', fg='#212529', state=tk.DISABLED
        )
        self.recording_stats.pack(fill=tk.BOTH, expand=True)
        
        # Playback control (for Mock Client)
        playback_group = ttk.LabelFrame(main_pane, text="Playback Control (Mock Client Only)", padding=15)
        main_pane.add(playback_group, weight=0)
        
        # Load recording file section
        load_frame = ttk.Frame(playback_group)
        load_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(load_frame, text="Recording File:").pack(side=tk.LEFT, padx=(0, 5))
        self.playback_file_path = tk.StringVar()
        playback_entry = ttk.Entry(load_frame, textvariable=self.playback_file_path, 
                                   width=40, state=tk.DISABLED)
        playback_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.load_playback_btn = ttk.Button(load_frame, text="Load File", 
                                           command=self.load_playback_file,
                                           state=tk.DISABLED, width=12)
        self.load_playback_btn.pack(side=tk.LEFT, padx=5)
        
        playback_btn_frame = ttk.Frame(playback_group)
        playback_btn_frame.pack(fill=tk.X, pady=10)
        
        self.playback_play_btn = ttk.Button(playback_btn_frame, text="Play", 
                                           command=self.playback_play,
                                           state=tk.DISABLED, width=12)
        self.playback_play_btn.pack(side=tk.LEFT, padx=5)
        
        self.playback_pause_btn = ttk.Button(playback_btn_frame, text="Pause", 
                                            command=self.playback_pause,
                                            state=tk.DISABLED, width=12)
        self.playback_pause_btn.pack(side=tk.LEFT, padx=5)
        
        self.playback_stop_btn = ttk.Button(playback_btn_frame, text="Stop", 
                                           command=self.playback_stop,
                                            state=tk.DISABLED, width=12)
        self.playback_stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Playback info
        playback_info_frame = ttk.Frame(playback_group)
        playback_info_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(playback_info_frame, text="Progress:").pack(side=tk.LEFT, padx=5)
        self.playback_progress_label = ttk.Label(playback_info_frame, text="0.0%", 
                                                 foreground="#6c757d")
        self.playback_progress_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(playback_info_frame, text="Time:").pack(side=tk.LEFT, padx=(20, 5))
        self.playback_time_label = ttk.Label(playback_info_frame, text="0.0s", 
                                             foreground="#6c757d")
        self.playback_time_label.pack(side=tk.LEFT, padx=5)
        
        # Update playback info periodically
        self.update_playback_info()
        
    def setup_update_loop(self):
        """Setup periodic update loop with separated image and status updates."""
        # Status update loop (slower, 1-2 Hz)
        def status_update_loop():
            while not self.stop_update.is_set():
                try:
                    if self.is_connected and self.client:
                        if hasattr(self, 'auto_refresh') and self.auto_refresh.get():
                            self.root.after(0, self.update_status_display)
                        if self.auto_update_pc.get() and HAS_MATPLOTLIB:
                            self.root.after(0, self.update_pointcloud_display)
                        # Update recording stats
                        if hasattr(self, 'start_record_btn') and self.start_record_btn['state'] == tk.DISABLED:
                            self.root.after(0, self.update_recording_stats)
                except Exception as e:
                    self.log(f"Status update error: {e}", "error")
                time.sleep(1.0)  # Status updates every second
                
        # Image update loop (faster, 30-60 FPS)
        def image_update_loop():
            while not self.stop_update.is_set():
                try:
                    # Check connection status more thoroughly
                    is_connected = (self.is_connected and 
                                   self.client and 
                                   self.client.is_connected() if self.client else False)
                    
                    if is_connected and self.auto_update_image.get() and HAS_CV2:
                        # Dynamically calculate frame rate based on current client type
                        target_fps = 30.0  # Default 30 FPS
                        if HAS_AIRSIM and isinstance(self.client, AirSimClient):
                            # AirSim can handle higher frame rates
                            target_fps = 60.0
                        frame_interval = 1.0 / target_fps
                        
                        # Update main image display
                        if not self._pending_image_update.get("main", False):
                            self.root.after(0, self.update_image_display)
                        
                        # Also update control tab image if AirSim client
                        if HAS_AIRSIM and isinstance(self.client, AirSimClient):
                            if not self._pending_image_update.get("control", False):
                                self.root.after(0, self.update_control_image_display)
                        
                        time.sleep(frame_interval)
                    else:
                        # If not connected, sleep longer to avoid busy waiting
                        time.sleep(0.1)
                except Exception as e:
                    # Log error but continue
                    error_str = str(e)
                    if "cannot be re-sized" not in error_str and "Existing exports" not in error_str:
                        try:
                            self.log(f"Image update loop error: {e}", "error")
                        except:
                            pass
                    time.sleep(0.1)
                
        self.update_thread = threading.Thread(target=status_update_loop, daemon=True)
        self.update_thread.start()
        
        self.image_update_thread = threading.Thread(target=image_update_loop, daemon=True)
        self.image_update_thread.start()
        
    def log(self, message: str, level: str = "info"):
        """Log message to connection status with color coding."""
        timestamp = time.strftime("%H:%M:%S")
        
        # Color mapping
        colors = {
            "info": "#212529",
            "success": "#28a745",
            "warning": "#ffc107",
            "error": "#dc3545",
        }
        
        color = colors.get(level, colors["info"])
        
        # Insert colored text
        self.connection_status.insert(tk.END, f"[{timestamp}] {message}\n")
        
        # Tag and apply color
        start = f"end-{len(message)+len(timestamp)+4}c"
        end = "end-1c"
        self.connection_status.tag_add(f"log_{level}", start, end)
        self.connection_status.tag_config(f"log_{level}", foreground=color)
        
        self.connection_status.see(tk.END)
        
        # Limit log length (keep last 1000 lines)
        lines = int(self.connection_status.index('end-1c').split('.')[0])
        if lines > 1000:
            self.connection_status.delete('1.0', f'{lines-1000}.0')
    
    def clear_log(self):
        """Clear connection log"""
        self.connection_status.delete('1.0', tk.END)
        self.log("Log cleared", "info")
    
    def save_log(self):
        """Save log to file"""
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.connection_status.get('1.0', tk.END))
                self.log(f"Log saved to: {filename}", "success")
                messagebox.showinfo("Success", "Log saved")
        except Exception as e:
            self.log(f"Failed to save log: {e}", "error")
    
    def clear_editor(self):
        """Clear message editor"""
        self.message_editor.delete('1.0', tk.END)
    
    def format_json(self):
        """Format JSON"""
        try:
            text = self.message_editor.get('1.0', tk.END).strip()
            if text:
                obj = json.loads(text)
                formatted = json.dumps(obj, indent=4, ensure_ascii=False)
                self.message_editor.delete('1.0', tk.END)
                self.message_editor.insert('1.0', formatted)
        except json.JSONDecodeError as e:
            messagebox.showerror("Error", f"JSON format error: {e}")
    
    def clear_history(self):
        """Clear command history"""
        self.command_history.delete('1.0', tk.END)
    
    def clear_test_results(self):
        """Clear test results"""
        self.test_results.delete('1.0', tk.END)
    
    def save_test_results(self):
        """Save test results"""
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.test_results.get('1.0', tk.END))
                messagebox.showinfo("Success", "Test results saved")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {e}")
        
    def update_connection_indicator(self, connected: bool):
        """Update connection status indicator"""
        self.status_indicator.delete("all")
        client_type_name = self.get_client_type_name()
        
        if connected:
            self.status_indicator.create_oval(4, 4, 12, 12, fill='#28a745', outline='')
            self.status_indicator_text.config(text="Connected", foreground='#28a745')
            self.client_type_label.config(text=f"Client: {client_type_name}", foreground='#28a745')
            if client_type_name == "AirSim":
                self.info_label.config(text="Connected to AirSim")
            elif client_type_name == "Mock":
                self.info_label.config(text="Connected to Mock Client")
            else:
                self.info_label.config(text="Connected to ROS server")
        else:
            self.status_indicator.create_oval(4, 4, 12, 12, fill='#dc3545', outline='')
            self.status_indicator_text.config(text="Disconnected", foreground='#dc3545')
            self.client_type_label.config(text="Client: None", foreground='#6c757d')
            self.info_label.config(text="Disconnected")
    
    def on_client_type_changed(self):
        """Handle client type selection change."""
        client_type = self.client_type.get()
        
        # Update connection label and placeholder
        if client_type == "airsim":
            self.connection_label.config(text="AirSim Address (ip:port):")
            self.connection_url.set("127.0.0.1:41451")
            self.recording_file_frame.pack_forget()
            self.use_mock.set(False)
        elif client_type == "mock":
            self.connection_label.config(text="Connection String:")
            self.connection_url.set("mock://test")
            self.recording_file_frame.pack(fill=tk.X, pady=8, before=self.connect_btn.master)
            self.use_mock.set(True)
        else:  # ros
            self.connection_label.config(text="WebSocket URL:")
            self.connection_url.set("ws://192.168.27.152:9090")
            self.recording_file_frame.pack_forget()
            self.use_mock.set(False)
        
        # Update quick connect buttons
        self.update_quick_connect_buttons()
    
    def update_quick_connect_buttons(self):
        """Update quick connect buttons based on client type."""
        # Clear existing buttons
        for widget in self.quick_buttons_frame.winfo_children():
            widget.destroy()
        
        client_type = self.client_type.get()
        if client_type == "airsim":
            quick_urls = self.airsim_quick_urls
        else:
            quick_urls = self.ros_quick_urls
        
        for i, (name, url) in enumerate(quick_urls):
            btn = ttk.Button(self.quick_buttons_frame, text=name, 
                           command=lambda u=url: self.connection_url.set(u),
                           width=12)
            btn.grid(row=i//2, column=i%2, padx=5, pady=3, sticky=tk.W+tk.E)
            self.quick_buttons_frame.columnconfigure(i%2, weight=1)
    
    def on_mock_mode_changed(self):
        """Handle mock mode checkbox change (legacy compatibility)."""
        if self.use_mock.get():
            self.recording_file_frame.pack(fill=tk.X, pady=8, before=self.connect_btn.master)
        else:
            self.recording_file_frame.pack_forget()
    
    def browse_recording_file(self):
        """Browse for recording file."""
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            title="Select Recording File",
            filetypes=[("ROS Recording", "*.rosrec"), ("All files", "*.*")]
        )
        if filename:
            self.recording_file_path.set(filename)
            self.log(f"Selected recording file: {filename}", "info")
    
    def connect(self):
        """Connect to ROS bridge, Mock client, or AirSim."""
        url = self.connection_url.get().strip()
        if not url:
            messagebox.showerror("Error", "Please enter connection address")
            return
        
        client_type = self.client_type.get()
        
        # Validate AirSim availability
        if client_type == "airsim" and not HAS_AIRSIM:
            messagebox.showerror("Error", "AirSim client is not available. Please install airsim package.")
            return
            
        try:
            self.log(f"Connecting to {url} ({client_type.upper()})...", "info")
            self.connect_btn.config(state=tk.DISABLED)
            self.info_label.config(text="Connecting...")
            
            def connect_thread():

                try:
                    # Use local variables to avoid closure issues
                    connection_url = url
                    conn_type = client_type
                    
                    if conn_type == "mock":
                        # Prepare config for Mock Client
                        config = {}
                        recording_file = self.recording_file_path.get().strip()
                        if recording_file:
                            config["playback_file"] = recording_file
                            config["playback_loop"] = True  # Auto loop when file is provided
                            self.root.after(0, lambda: self.log(f"Using Mock client with recording: {recording_file}", "info"))
                        else:
                            self.root.after(0, lambda: self.log("Using Mock client (Test Mode)", "warning"))
                        
                        self.client = MockRosClient(connection_url, config=config)
                        self.client.connect_async()
                        
                        # Enable playback controls if in playback mode
                        if recording_file and self.client.is_playback_mode():
                            self.root.after(0, lambda: self.enable_playback_controls())
                    elif conn_type == "airsim":
                        # AirSim connection
                        # Remove ws:// or http:// prefix if present
                        clean_url = connection_url.replace("ws://", "").replace("http://", "").replace("https://", "")
                        self.root.after(0, lambda u=clean_url: self.log(f"Connecting to AirSim at {u}...", "info"))
                        
                        self.client = AirSimClient(clean_url)
                        self.client.connect_async()
                    else:  # ros
                        # ROS connection
                        ros_url = connection_url
                        if not ros_url.startswith("ws://") and not ros_url.startswith("wss://"):
                            ros_url = "ws://" + ros_url
                        self.client = RosClient(ros_url)
                        self.client.connect_async()
                        
                    # Wait a bit for connection (longer for AirSim)
                    wait_time = 3 if conn_type == "airsim" else 2
                    time.sleep(wait_time)
                    
                    # Check connection with retries for AirSim
                    max_retries = 5 if conn_type == "airsim" else 1
                    connected = False
                    for _ in range(max_retries):
                        if self.client.is_connected():
                            connected = True
                            break
                        time.sleep(1)
                    
                    if connected:
                        self.is_connected = True
                        self.root.after(0, lambda: self.connect_btn.config(state=tk.DISABLED))
                        self.root.after(0, lambda: self.disconnect_btn.config(state=tk.NORMAL))
                        client_name = "AirSim" if conn_type == "airsim" else ("Mock" if conn_type == "mock" else "ROS")
                        self.root.after(0, lambda name=client_name: self.log(f"Connection successful! ({name})", "success"))
                        self.root.after(0, lambda: self.update_connection_indicator(True))
                        
                        # Update control UI based on client type
                        self.root.after(0, lambda: self.update_control_ui_for_client())
                        
                        # Enable playback controls if Mock Client
                        if isinstance(self.client, MockRosClient):
                            self.root.after(0, lambda: self.load_playback_btn.config(state=tk.NORMAL))
                            if self.client.is_playback_mode():
                                self.root.after(0, lambda: self.enable_playback_controls())
                                self.root.after(0, lambda: self.playback_file_path.set(self.client._playback_file or ""))
                    else:
                        error_msg = f"Connection failed, please check {'AirSim server' if conn_type == 'airsim' else 'URL and network'}"
                        self.root.after(0, lambda msg=error_msg: self.log(msg, "error"))
                        self.root.after(0, lambda msg=error_msg: messagebox.showwarning("Warning", msg))
                        self.root.after(0, lambda: self.connect_btn.config(state=tk.NORMAL))
                        self.root.after(0, lambda: self.info_label.config(text="Connection failed"))
                except Exception as e:
                    error_msg = str(e)
                    self.root.after(0, lambda msg=error_msg: self.log(f"Connection error: {msg}", "error"))
                    self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Connection failed: {msg}"))
                    self.root.after(0, lambda: self.connect_btn.config(state=tk.NORMAL))
                    self.root.after(0, lambda: self.info_label.config(text="Connection error"))
            
            threading.Thread(target=connect_thread, daemon=True).start()
                
        except Exception as e:
            self.log(f"Connection error: {e}", "error")
            messagebox.showerror("Error", f"Connection failed: {e}")
            self.connect_btn.config(state=tk.NORMAL)
            
    def disconnect(self):
        """Disconnect from ROS bridge."""
        try:
            # Stop recording if active
            if self.client and self.client.is_recording():
                self.client.stop_recording()
            
            if self.client:
                self.client.terminate()
                self.log("Disconnected", "info")
            self.is_connected = False
            self.client = None
            self.connect_btn.config(state=tk.NORMAL)
            self.disconnect_btn.config(state=tk.DISABLED)
            self.update_connection_indicator(False)
            
            # Update control UI (hide AirSim actions)
            self.update_control_ui_for_client()
            
            # Disable recording controls
            self.start_record_btn.config(state=tk.NORMAL)
            self.stop_record_btn.config(state=tk.DISABLED)
            self.save_record_btn.config(state=tk.DISABLED)
            
            # Disable playback controls
            self.disable_playback_controls()
            self.load_playback_btn.config(state=tk.DISABLED)
            self.playback_file_path.set("")
            
            # Clear status display
            for label in self.status_labels.values():
                label.config(text="N/A", foreground="#6c757d")
            
            # Clear image display state
            self._last_canvas_size.clear()
            self._pending_image_update = {"main": False, "control": False}
        except Exception as e:
            self.log(f"Disconnect error: {e}", "error")
            
    def update_status_display(self):
        """Update status display."""
        if not self.client or not self.is_connected:
            return
            
        try:
            # Check if client is actually connected
            if not self.client.is_connected():
                return
                
            state = self.client.get_status()
            pos = self.client.get_position()
            ori = self.client.get_orientation()
            
            status_data = {
                "connected": ("Connected", "#28a745") if state.connected else ("Disconnected", "#dc3545"),
                "armed": ("Armed", "#28a745") if state.armed else ("Disarmed", "#6c757d"),
                "mode": (state.mode or "N/A", "#17a2b8"),
                "battery": (f"{state.battery:.1f}%", "#28a745" if state.battery > 20 else "#ffc107" if state.battery > 10 else "#dc3545"),
                "latitude": (f"{pos[0]:.6f}", "#212529"),
                "longitude": (f"{pos[1]:.6f}", "#212529"),
                "altitude": (f"{pos[2]:.2f}m", "#212529"),
                "roll": (f"{ori[0]:.2f}deg", "#212529"),
                "pitch": (f"{ori[1]:.2f}deg", "#212529"),
                "yaw": (f"{ori[2]:.2f}deg", "#212529"),
                "landed": ("Landed", "#6c757d") if state.landed else ("Flying", "#28a745"),
                "reached": ("Yes", "#28a745") if state.reached else ("No", "#6c757d"),
                "returned": ("Yes", "#28a745") if state.returned else ("No", "#6c757d"),
                "tookoff": ("Yes", "#28a745") if state.tookoff else ("No", "#6c757d"),
            }
            
            for field, label in self.status_labels.items():
                if field in status_data:
                    value, color = status_data[field]
                    label.config(text=value, foreground=color)
                else:
                    label.config(text="N/A", foreground="#6c757d")
                
        except Exception as e:
            # Only log if it's not a transient error
            error_str = str(e)
            if "cannot be re-sized" not in error_str and "Existing exports" not in error_str:
                self.log(f"状态更新错误: {e}", "error")
            
    def update_image_display(self):
        """Update image display with frame skipping optimization."""
        if not self.client:
            return
        
        # Check both GUI flag and client connection status
        if not self.is_connected or not self.client.is_connected():
            return
        
        # Skip if update is already pending (prevent queue buildup)
        if self._pending_image_update.get("main", False):
            return
            
        try:
            # Try to get latest image (non-blocking)
            image_data = self.client.get_latest_image()
            if image_data:
                frame, timestamp = image_data
                self._pending_image_update["main"] = True
                
                # Use after_idle for smoother updates
                def update():
                    try:
                        self.display_image(frame, cache_key="main")
                        # Show timestamp to verify updates are happening
                        info_text = f"Size: {frame.shape[1]}x{frame.shape[0]}, Time: {timestamp:.3f}"
                        if hasattr(self, 'image_info_label'):
                            self.image_info_label.config(text=info_text, foreground="#212529")
                    finally:
                        self._pending_image_update["main"] = False
                
                self.root.after_idle(update)
            else:
                # No image available - this is normal if image update thread hasn't started yet
                pass
        except Exception as e:
            self._pending_image_update["main"] = False
            # Only log non-transient errors
            error_str = str(e)
            if "cannot be re-sized" not in error_str and "Existing exports" not in error_str:
                # Log error for debugging
                try:
                    self.log(f"Image update error: {e}", "error")
                except:
                    pass
            
    def display_image(self, frame, canvas=None, cache_key="main"):
        """
        Display image on canvas with optimization.
        
        Args:
            frame: Image frame (numpy array)
            canvas: Target canvas (default: self.image_canvas)
            cache_key: Cache key for this canvas
        """
        if canvas is None:
            canvas = self.image_canvas
            
        try:
            # Get canvas dimensions
            canvas.update_idletasks()  # Ensure canvas is rendered
            canvas_width = max(canvas.winfo_width(), 1)
            canvas_height = max(canvas.winfo_height(), 1)
            
            # Always process and display the current frame (no content caching)
            # Cache is only used to avoid unnecessary resizing when canvas size is stable
            
            # Calculate resize dimensions
            if canvas_width > 1 and canvas_height > 1:
                scale = min(canvas_width / frame.shape[1], canvas_height / frame.shape[0])
                new_width = int(frame.shape[1] * scale)
                new_height = int(frame.shape[0] * scale)
                
                # Check if we need to resize (canvas size changed)
                current_size = (canvas_width, canvas_height)
                last_size = self._last_canvas_size.get(cache_key, (0, 0))
                
                # Only resize if canvas size changed
                if current_size != last_size or new_width != frame.shape[1] or new_height != frame.shape[0]:
                    # Use faster interpolation for real-time display
                    frame_resized = cv2.resize(frame, (new_width, new_height), 
                                              interpolation=cv2.INTER_LINEAR)
                else:
                    # Canvas size unchanged, but still need to process the new frame
                    frame_resized = frame
                    if new_width != frame.shape[1] or new_height != frame.shape[0]:
                        frame_resized = cv2.resize(frame, (new_width, new_height), 
                                                  interpolation=cv2.INTER_LINEAR)
            else:
                frame_resized = frame
            
            # Convert to RGB based on client type (only if needed)
            if HAS_AIRSIM and isinstance(self.client, AirSimClient):
                # AirSim already provides RGB format
                frame_rgb = frame_resized
            else:
                # ROS and Mock clients return BGR format
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image and PhotoImage (always create new for new frame)
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image=image)
            
            # Update canvas (always update to show new frame)
            canvas.delete("all")
            canvas.create_image(
                canvas_width // 2, canvas_height // 2,
                image=photo, anchor=tk.CENTER
            )
            canvas.image = photo  # Keep reference to prevent garbage collection
            
            # Update canvas size tracking
            self._last_canvas_size[cache_key] = (canvas_width, canvas_height)
            
        except Exception as e:
            # Silently fail to avoid spam
            pass
            
    def update_control_image_display(self):
        """Update image display in control tab (for AirSim) with optimization."""
        if not self.client:
            return
        
        # Check both GUI flag and client connection status
        if not self.is_connected or not self.client.is_connected():
            return
            
        if not HAS_CV2 or not hasattr(self, 'control_image_canvas'):
            return
        if not (HAS_AIRSIM and isinstance(self.client, AirSimClient)):
            return
        
        # Skip if update is already pending
        if self._pending_image_update.get("control", False):
            return
            
        try:
            # Try to get latest image (non-blocking)
            image_data = self.client.get_latest_image()
            if image_data:
                frame, timestamp = image_data
                self._pending_image_update["control"] = True
                
                # Use after_idle for smoother updates
                def update():
                    try:
                        self.display_control_image(frame)
                        # Show timestamp to verify updates are happening
                        info_text = f"Size: {frame.shape[1]}x{frame.shape[0]}, Time: {timestamp:.3f}"
                        if hasattr(self, 'control_image_info'):
                            self.control_image_info.config(text=info_text, foreground="#212529")
                    finally:
                        self._pending_image_update["control"] = False
                
                self.root.after_idle(update)
            else:
                # No image available - this is normal if image update thread hasn't started yet
                pass
        except Exception as e:
            self._pending_image_update["control"] = False
            # Only log non-transient errors
            error_str = str(e)
            if "cannot be re-sized" not in error_str and "Existing exports" not in error_str:
                # Log error for debugging
                try:
                    self.log(f"Control image update error: {e}", "error")
                except:
                    pass
    
    def display_control_image(self, frame):
        """Display image on control tab canvas with optimization."""
        if not HAS_CV2 or not hasattr(self, 'control_image_canvas'):
            return
        
        # Use optimized display_image method
        self.display_image(frame, canvas=self.control_image_canvas, cache_key="control")
    
    def fetch_image_manual(self):
        """Manually fetch image."""
        if not self.client or not self.is_connected:
            messagebox.showwarning("Warning", "Please connect first")
            return
            
        try:
            self.log("Fetching image...", "info")
            self.image_info_label.config(text="Fetching image...")
            image_data = self.client.fetch_camera_image()
            if image_data:
                frame, timestamp = image_data
                self.display_image(frame)
                info_text = f"Timestamp: {timestamp:.3f}, Size: {frame.shape[1]}x{frame.shape[0]}"
                self.image_info_label.config(text=info_text, foreground="#28a745")
                self.log("Image fetched successfully", "success")
            else:
                self.image_info_label.config(text="No image received", foreground="#dc3545")
                self.log("No image received", "warning")
                messagebox.showinfo("Info", "No image data received")
        except Exception as e:
            self.image_info_label.config(text=f"Fetch failed: {e}", foreground="#dc3545")
            self.log(f"Image fetch error: {e}", "error")
            messagebox.showerror("Error", f"Failed to fetch image: {e}")
            
    def update_pointcloud_display(self):
        """Update point cloud display."""
        if not self.client or not self.is_connected:
            return
            
        try:
            # Try to get latest point cloud
            if hasattr(self.client, 'get_latest_point_cloud'):
                pc_data = self.client.get_latest_point_cloud()
                if pc_data:
                    points, timestamp = pc_data
                    self.display_pointcloud(points)
                    info_text = f"Timestamp: {timestamp:.3f}, Points: {len(points)}"
                    self.pc_info_label.config(text=info_text, foreground="#212529")
        except Exception as e:
            pass  # Silently fail
            
    def display_pointcloud(self, points):
        """Display point cloud."""
        try:
            self.pc_ax.clear()
            
            if len(points) > 0:
                # Sample points if too many
                original_count = len(points)
                if len(points) > 10000:
                    indices = np.random.choice(len(points), 10000, replace=False)
                    points = points[indices]
                    
                self.pc_ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                                 c=points[:, 2], cmap='viridis', s=1, alpha=0.6)
                self.pc_ax.set_xlabel('X (m)', fontsize=9)
                self.pc_ax.set_ylabel('Y (m)', fontsize=9)
                self.pc_ax.set_zlabel('Z (m)', fontsize=9)
                title = f'Point Cloud Visualization ({original_count} points)'
                if original_count > 10000:
                    title += f' (showing {len(points)} points)'
                self.pc_ax.set_title(title, fontsize=11, pad=10)
                
            self.pc_canvas.draw()
        except Exception as e:
            self.log(f"Point cloud display error: {e}", "error")
            
    def fetch_pointcloud_manual(self):
        """Manually fetch point cloud."""
        if not self.client or not self.is_connected:
            messagebox.showwarning("Warning", "Please connect first")
            return
            
        try:
            self.log("Fetching point cloud...", "info")
            self.pc_info_label.config(text="Fetching point cloud...")
            pc_data = self.client.fetch_point_cloud()
            if pc_data:
                points, timestamp = pc_data
                self.display_pointcloud(points)
                info_text = f"Timestamp: {timestamp:.3f}, Points: {len(points)}"
                self.pc_info_label.config(text=info_text, foreground="#28a745")
                self.log(f"Point cloud fetched successfully, points: {len(points)}", "success")
            else:
                self.pc_info_label.config(text="No point cloud received", foreground="#dc3545")
                self.log("No point cloud received", "warning")
                messagebox.showinfo("Info", "No point cloud data received")
        except Exception as e:
            self.pc_info_label.config(text=f"Fetch failed: {e}", foreground="#dc3545")
            self.log(f"Point cloud fetch error: {e}", "error")
            messagebox.showerror("Error", f"Failed to fetch point cloud: {e}")
            
    def set_preset_command(self, command: str):
        """Set preset command in editor."""
        self.message_editor.delete("1.0", tk.END)
        self.message_editor.insert("1.0", command)
        
    def send_control_command(self):
        """Send control command."""
        if not self.client or not self.is_connected:
            messagebox.showwarning("Warning", "Please connect first")
            return
            
        try:
            topic = self.control_topic.get().strip()
            topic_type = self.control_type.get().strip()
            message_text = self.message_editor.get("1.0", tk.END).strip()
            
            if not topic or not message_text:
                messagebox.showwarning("Warning", "Please fill in Topic and message content")
                return
                
            # Parse JSON
            try:
                message = json.loads(message_text)
            except json.JSONDecodeError as e:
                messagebox.showerror("Error", f"JSON format error: {e}")
                return
                
            # Send command
            self.client.publish(topic, topic_type, message)
            
            # Log to history with formatting
            timestamp = time.strftime("%H:%M:%S")
            history_text = f"[{timestamp}] {topic} ({topic_type}):\n{json.dumps(message, indent=2, ensure_ascii=False)}\n{'-'*60}\n"
            self.command_history.insert(tk.END, history_text)
            self.command_history.see(tk.END)
            
            self.log(f"Command sent: {topic} -> {message_text[:50]}...", "success")
            self.info_label.config(text=f"Command sent to {topic}")
            
        except Exception as e:
            self.log(f"Send command error: {e}", "error")
            messagebox.showerror("Error", f"Failed to send command: {e}")
            
    def test_connection(self):
        """Test ROS connection."""
        url = self.test_url.get().strip()
        timeout = float(self.test_timeout.get() or "5")
        
        self.test_results.insert(tk.END, f"\n{'='*60}\n")
        self.test_results.insert(tk.END, f"Starting connection test: {url}\n")
        self.test_results.see(tk.END)
        
        def run_test():
            try:
                test_client = RosClient(url)
                test_client.connect_async()
                
                # Wait for connection
                start_time = time.time()
                while time.time() - start_time < timeout:
                    if test_client.is_connected():
                        self.root.after(0, lambda: self.test_results.insert(tk.END, f"Connection successful!\n"))
                        test_client.terminate()
                        self.root.after(0, lambda: self.test_results.see(tk.END))
                        return
                    time.sleep(0.5)
                    
                self.root.after(0, lambda: self.test_results.insert(tk.END, f"Connection timeout\n"))
                test_client.terminate()
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: self.test_results.insert(tk.END, f"Connection failed: {msg}\n"))
            finally:
                self.root.after(0, lambda: self.test_results.see(tk.END))
                
        threading.Thread(target=run_test, daemon=True).start()
        
    def test_topics(self):
        """Test ROS topics."""
        if not self.client or not self.is_connected:
            messagebox.showwarning("Warning", "Please connect to ROS first")
            return
            
        self.test_results.insert(tk.END, f"\n{'='*60}\n")
        self.test_results.insert(tk.END, "Starting Topic test...\n")
        self.test_results.see(tk.END)
        
        # Test common topics
        topics_to_test = [
            "/mavros/state",
            "/mavros/battery",
            "/mavros/global_position/global",
        ]
        
        for topic in topics_to_test:
            try:
                # This is a simplified test - in real scenario you'd subscribe
                self.test_results.insert(tk.END, f"  - {topic}: Requires manual subscription test\n")
            except Exception as e:
                self.test_results.insert(tk.END, f"  - {topic}: Failed {e}\n")
                
        self.test_results.see(tk.END)
        
    def run_full_test(self):
        """Run full network test."""
        url = self.test_url.get().strip()
        timeout = float(self.test_timeout.get() or "5")
        
        self.test_results.insert(tk.END, f"\n{'='*60}\n")
        self.test_results.insert(tk.END, f"Starting full test: {url}\n")
        self.test_results.insert(tk.END, f"{'='*60}\n")
        self.test_results.see(tk.END)
        
        def run_full():
            try:
                # Test 1: Connection
                self.root.after(0, lambda: self.test_results.insert(tk.END, "\n[1/3] Testing connection...\n"))
                test_client = RosClient(url)
                test_client.connect_async()
                
                start_time = time.time()
                connected = False
                while time.time() - start_time < timeout:
                    if test_client.is_connected():
                        connected = True
                        self.root.after(0, lambda: self.test_results.insert(tk.END, "  Connection successful\n"))
                        break
                    time.sleep(0.5)
                    
                if not connected:
                    self.root.after(0, lambda: self.test_results.insert(tk.END, "  Connection failed\n"))
                    test_client.terminate()
                    self.root.after(0, lambda: self.test_results.see(tk.END))
                    return
                    
                # Test 2: Status
                self.root.after(0, lambda: self.test_results.insert(tk.END, "\n[2/3] Testing status retrieval...\n"))
                try:
                    state = test_client.get_status()
                    self.root.after(0, lambda: self.test_results.insert(tk.END, f"  Status retrieval successful\n"))
                    self.root.after(0, lambda: self.test_results.insert(tk.END, f"    Mode: {state.mode}, Battery: {state.battery}%\n"))
                except Exception as e:
                    error_msg = str(e)
                    self.root.after(0, lambda msg=error_msg: self.test_results.insert(tk.END, f"  Status retrieval failed: {msg}\n"))
                    
                # Test 3: Publish
                self.root.after(0, lambda: self.test_results.insert(tk.END, "\n[3/3] Testing message publish...\n"))
                try:
                    test_client.publish("/test", "std_msgs/String", {"data": "test"})
                    self.root.after(0, lambda: self.test_results.insert(tk.END, "  Message publish successful\n"))
                except Exception as e:
                    error_msg = str(e)
                    self.root.after(0, lambda msg=error_msg: self.test_results.insert(tk.END, f"  Message publish failed: {msg}\n"))
                    
                test_client.terminate()
                self.root.after(0, lambda: self.test_results.insert(tk.END, f"\n{'='*60}\n"))
                self.root.after(0, lambda: self.test_results.insert(tk.END, "Test completed!\n"))
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: self.test_results.insert(tk.END, f"\nTest error: {msg}\n"))
            finally:
                self.root.after(0, lambda: self.test_results.see(tk.END))
                
        threading.Thread(target=run_full, daemon=True).start()
    
    # ---------- Recording control methods ----------
    
    def start_recording(self):
        """Start recording."""
        if not self.client or not self.is_connected:
            messagebox.showwarning("Warning", "Please connect first")
            return
        
        try:
            success = self.client.start_recording(
                record_images=self.record_images.get(),
                record_pointclouds=self.record_pointclouds.get(),
                record_states=self.record_states.get(),
                image_quality=85
            )
            if success:
                self.start_record_btn.config(state=tk.DISABLED)
                self.stop_record_btn.config(state=tk.NORMAL)
                self.save_record_btn.config(state=tk.DISABLED)
                self.recording_stats.config(state=tk.NORMAL)
                self.recording_stats.delete('1.0', tk.END)
                self.recording_stats.insert('1.0', "Recording started...\n")
                self.recording_stats.config(state=tk.DISABLED)
                # Note: client.start_recording() already logs "Recording started", so we don't log again here
            else:
                messagebox.showerror("Error", "Failed to start recording")
        except Exception as e:
            self.log(f"Start recording error: {e}", "error")
            messagebox.showerror("Error", f"Failed to start recording: {e}")
    
    def stop_recording(self):
        """Stop recording."""
        if not self.client:
            return
        
        try:
            success = self.client.stop_recording()
            if success:
                self.start_record_btn.config(state=tk.NORMAL)
                self.stop_record_btn.config(state=tk.DISABLED)
                self.save_record_btn.config(state=tk.NORMAL)
                # Note: client.stop_recording() already logs "Recording stopped", so we don't log again here
                self.update_recording_stats()
            else:
                messagebox.showwarning("Warning", "No active recording to stop")
        except Exception as e:
            self.log(f"Stop recording error: {e}", "error")
            messagebox.showerror("Error", f"Failed to stop recording: {e}")
    
    def save_recording(self):
        """Save recording to file."""
        if not self.client:
            messagebox.showwarning("Warning", "No client available")
            return
        
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                title="Save Recording",
                defaultextension=".rosrec",
                filetypes=[("ROS Recording", "*.rosrec"), ("All files", "*.*")]
            )
            if filename:
                success = self.client.save_recording(filename, compress=True)
                if success:
                    self.log(f"Recording saved to: {filename}", "success")
                    messagebox.showinfo("Success", f"Recording saved to:\n{filename}")
                else:
                    messagebox.showerror("Error", "Failed to save recording")
        except Exception as e:
            self.log(f"Save recording error: {e}", "error")
            messagebox.showerror("Error", f"Failed to save recording: {e}")
    
    def update_recording_stats(self):
        """Update recording statistics display."""
        if not self.client:
            return
        
        try:
            stats = self.client.get_recording_statistics()
            if stats:
                self.recording_stats.config(state=tk.NORMAL)
                self.recording_stats.delete('1.0', tk.END)
                
                stats_text = f"Recording Statistics:\n"
                stats_text += f"{'='*50}\n"
                stats_text += f"Images recorded: {stats.get('images_recorded', 0)}\n"
                stats_text += f"Point clouds recorded: {stats.get('pointclouds_recorded', 0)}\n"
                stats_text += f"States recorded: {stats.get('states_recorded', 0)}\n"
                stats_text += f"Total entries: {stats.get('total_entries', 0)}\n"
                stats_text += f"Dropped: {stats.get('dropped', 0)}\n"
                stats_text += f"Queue size: {stats.get('queue_size', 0)}\n"
                stats_text += f"Status: {'Recording' if stats.get('is_recording', False) else 'Stopped'}\n"
                
                self.recording_stats.insert('1.0', stats_text)
                self.recording_stats.config(state=tk.DISABLED)
        except Exception as e:
            pass  # Silently fail
    
    # ---------- Playback control methods ----------
    
    def enable_playback_controls(self):
        """Enable playback control buttons."""
        self.playback_play_btn.config(state=tk.NORMAL)
        self.playback_pause_btn.config(state=tk.NORMAL)
        self.playback_stop_btn.config(state=tk.NORMAL)
    
    def disable_playback_controls(self):
        """Disable playback control buttons."""
        self.playback_play_btn.config(state=tk.DISABLED)
        self.playback_pause_btn.config(state=tk.DISABLED)
        self.playback_stop_btn.config(state=tk.DISABLED)
        self.playback_progress_label.config(text="0.0%")
        self.playback_time_label.config(text="0.0s")
    
    def load_playback_file(self):
        """Load a recording file for playback (after connection)."""
        if not self.client:
            messagebox.showwarning("Warning", "Please connect first")
            return
        if not isinstance(self.client, MockRosClient):
            messagebox.showwarning("Warning", "Playback is only available with Mock Client")
            return
        
        try:
            from tkinter import filedialog
            filename = filedialog.askopenfilename(
                title="Select Recording File",
                filetypes=[("ROS Recording", "*.rosrec"), ("All files", "*.*")]
            )
            if filename:
                self.log(f"Loading recording file: {filename}", "info")
                success = self.client.load_recording_file(filename, auto_play=True, loop=True)
                if success:
                    self.playback_file_path.set(filename)
                    self.enable_playback_controls()
                    self.log("Recording file loaded and playback started", "success")
                    messagebox.showinfo("Success", f"Recording file loaded:\n{filename}\n\nPlayback started automatically.")
                else:
                    messagebox.showerror("Error", "Failed to load recording file")
        except Exception as e:
            self.log(f"Load playback file error: {e}", "error")
            messagebox.showerror("Error", f"Failed to load recording file: {e}")
    
    def playback_play(self):
        """Start or resume playback."""
        if not self.client:
            return
        if not isinstance(self.client, MockRosClient):
            messagebox.showwarning("Warning", "Playback is only available with Mock Client")
            return
        
        try:
            if self.client.playback_is_playing():
                messagebox.showinfo("Info", "Playback is already running")
                return
            
            success = self.client.playback_play()
            if success:
                self.log("Playback started/resumed", "info")
            else:
                messagebox.showwarning("Warning", "Failed to start playback")
        except Exception as e:
            self.log(f"Playback play error: {e}", "error")
            messagebox.showerror("Error", f"Failed to start playback: {e}")
    
    def playback_pause(self):
        """Pause playback."""
        if not self.client:
            return
        if not isinstance(self.client, MockRosClient):
            return
        
        try:
            success = self.client.playback_pause()
            if success:
                self.log("Playback paused", "info")
        except Exception as e:
            self.log(f"Playback pause error: {e}", "error")
    
    def playback_stop(self):
        """Stop playback."""
        if not self.client:
            return
        if not isinstance(self.client, MockRosClient):
            return
        
        try:
            success = self.client.playback_stop()
            if success:
                self.log("Playback stopped", "info")
        except Exception as e:
            self.log(f"Playback stop error: {e}", "error")
    
    def update_playback_info(self):
        """Update playback information display."""
        try:
            if self.client and isinstance(self.client, MockRosClient) and self.client.is_playback_mode():
                progress = self.client.playback_get_progress()
                current_time = self.client.playback_get_current_time()
                
                self.playback_progress_label.config(text=f"{progress*100:.1f}%")
                self.playback_time_label.config(text=f"{current_time:.1f}s")
            else:
                self.playback_progress_label.config(text="0.0%")
                self.playback_time_label.config(text="0.0s")
        except Exception:
            pass
        
        # Schedule next update
        self.root.after(500, self.update_playback_info)
    
    def get_client_type_name(self) -> str:
        """Get the name of the current client type."""
        if not self.client:
            return "None"
        if isinstance(self.client, MockRosClient):
            return "Mock"
        elif HAS_AIRSIM and isinstance(self.client, AirSimClient):
            return "AirSim"
        else:
            return "ROS"
    
    # ---------- AirSim Quick Actions ----------
    
    def update_control_ui_for_client(self):
        """Update control UI based on client type."""
        if not hasattr(self, 'control_main_pane'):
            return
        
        parent_pane = self.control_main_pane
        
        if HAS_AIRSIM and self.client and isinstance(self.client, AirSimClient):
            # Show AirSim actions, hide ROS/Mock presets
            try:
                # Remove preset group from pane
                parent_pane.forget(self.preset_group)
            except:
                pass
            
            # Add AirSim actions to pane (if not already added)
            try:
                parent_pane.add(self.airsim_actions_group, weight=0)
            except:
                # Already added, do nothing
                pass
        else:
            # Show ROS/Mock presets, hide AirSim actions
            try:
                # Remove AirSim actions from pane
                parent_pane.forget(self.airsim_actions_group)
            except:
                pass
            
            # Add preset group to pane (if not already added)
            try:
                parent_pane.add(self.preset_group, weight=0)
            except:
                # Already added, do nothing
                pass
    
    def airsim_arm(self):
        """Arm AirSim drone."""
        if not self.client or not self.is_connected:
            messagebox.showwarning("Warning", "Please connect to AirSim first")
            return
        if not isinstance(self.client, AirSimClient):
            messagebox.showwarning("Warning", "This action is only available for AirSim client")
            return
        
        # Run in background thread to avoid freezing GUI
        def arm_thread():
            try:
                self.root.after(0, lambda: self.log("AirSim: Arming...", "info"))
                success = self.client.arm(True)
                if success:
                    self.root.after(0, lambda: self.log("AirSim: Armed", "success"))
                    self.root.after(0, lambda: self.info_label.config(text="AirSim: Armed"))
                else:
                    self.root.after(0, lambda: self.log("AirSim: Arm failed", "error"))
                    self.root.after(0, lambda: messagebox.showerror("Error", "Failed to arm"))
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: self.log(f"AirSim arm error: {msg}", "error"))
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Failed to arm: {msg}"))
        
        threading.Thread(target=arm_thread, daemon=True).start()
    
    def airsim_disarm(self):
        """Disarm AirSim drone."""
        if not self.client or not self.is_connected:
            messagebox.showwarning("Warning", "Please connect to AirSim first")
            return
        if not isinstance(self.client, AirSimClient):
            messagebox.showwarning("Warning", "This action is only available for AirSim client")
            return
        
        # Run in background thread to avoid freezing GUI
        def disarm_thread():
            try:
                self.root.after(0, lambda: self.log("AirSim: Disarming...", "info"))
                success = self.client.arm(False)
                if success:
                    self.root.after(0, lambda: self.log("AirSim: Disarmed", "success"))
                    self.root.after(0, lambda: self.info_label.config(text="AirSim: Disarmed"))
                else:
                    self.root.after(0, lambda: self.log("AirSim: Disarm failed", "error"))
                    self.root.after(0, lambda: messagebox.showerror("Error", "Failed to disarm"))
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: self.log(f"AirSim disarm error: {msg}", "error"))
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Failed to disarm: {msg}"))
        
        threading.Thread(target=disarm_thread, daemon=True).start()
    
    def airsim_takeoff(self):
        """Takeoff AirSim drone."""
        if not self.client or not self.is_connected:
            messagebox.showwarning("Warning", "Please connect to AirSim first")
            return
        if not isinstance(self.client, AirSimClient):
            messagebox.showwarning("Warning", "This action is only available for AirSim client")
            return
        
        # Run in background thread to avoid freezing GUI
        def takeoff_thread():
            try:
                self.root.after(0, lambda: self.log("AirSim: Taking off...", "info"))
                self.root.after(0, lambda: self.info_label.config(text="AirSim: Taking off..."))
                success = self.client.takeoff()
                if success:
                    self.root.after(0, lambda: self.log("AirSim: Takeoff command sent", "success"))
                    self.root.after(0, lambda: self.info_label.config(text="AirSim: Taking off..."))
                else:
                    self.root.after(0, lambda: self.log("AirSim: Takeoff failed", "error"))
                    self.root.after(0, lambda: messagebox.showerror("Error", "Failed to takeoff"))
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: self.log(f"AirSim takeoff error: {msg}", "error"))
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Failed to takeoff: {msg}"))
        
        threading.Thread(target=takeoff_thread, daemon=True).start()
    
    def airsim_land(self):
        """Land AirSim drone."""
        if not self.client or not self.is_connected:
            messagebox.showwarning("Warning", "Please connect to AirSim first")
            return
        if not isinstance(self.client, AirSimClient):
            messagebox.showwarning("Warning", "This action is only available for AirSim client")
            return
        
        # Run in background thread to avoid freezing GUI
        def land_thread():
            try:
                self.root.after(0, lambda: self.log("AirSim: Landing...", "info"))
                self.root.after(0, lambda: self.info_label.config(text="AirSim: Landing..."))
                success = self.client.land()
                if success:
                    self.root.after(0, lambda: self.log("AirSim: Land command sent", "success"))
                    self.root.after(0, lambda: self.info_label.config(text="AirSim: Landing..."))
                else:
                    self.root.after(0, lambda: self.log("AirSim: Land failed", "error"))
                    self.root.after(0, lambda: messagebox.showerror("Error", "Failed to land"))
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: self.log(f"AirSim land error: {msg}", "error"))
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Failed to land: {msg}"))
        
        threading.Thread(target=land_thread, daemon=True).start()
    
    def airsim_move_to(self):
        """Move AirSim drone to specified position."""
        if not self.client or not self.is_connected:
            messagebox.showwarning("Warning", "Please connect to AirSim first")
            return
        if not isinstance(self.client, AirSimClient):
            messagebox.showwarning("Warning", "This action is only available for AirSim client")
            return
        
        # Get values in main thread (before starting background thread)
        try:
            x = float(self.airsim_x.get())
            y = float(self.airsim_y.get())
            z = float(self.airsim_z.get())
            velocity = float(self.airsim_velocity.get())
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
            return
        
        # Run in background thread to avoid freezing GUI
        def move_thread():
            try:
                self.root.after(0, lambda: self.log(f"AirSim: Moving to ({x}, {y}, {z})...", "info"))
                self.root.after(0, lambda: self.info_label.config(text=f"AirSim: Moving to ({x}, {y}, {z})"))
                success = self.client.move_to(x, y, z, velocity)
                if success:
                    self.root.after(0, lambda: self.log(f"AirSim: Moving to ({x}, {y}, {z}) at {velocity} m/s", "success"))
                    self.root.after(0, lambda: self.info_label.config(text=f"AirSim: Moving to ({x}, {y}, {z})"))
                else:
                    self.root.after(0, lambda: self.log("AirSim: Move failed", "error"))
                    self.root.after(0, lambda: messagebox.showerror("Error", "Failed to move to position"))
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: self.log(f"AirSim move_to error: {msg}", "error"))
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Failed to move: {msg}"))
        
        threading.Thread(target=move_thread, daemon=True).start()
    
    def airsim_set_velocity(self):
        """Set AirSim drone velocity."""
        if not self.client or not self.is_connected:
            messagebox.showwarning("Warning", "Please connect to AirSim first")
            return
        if not isinstance(self.client, AirSimClient):
            messagebox.showwarning("Warning", "This action is only available for AirSim client")
            return
        
        # Get values in main thread (before starting background thread)
        try:
            vx = float(self.airsim_vx.get())
            vy = float(self.airsim_vy.get())
            vz = float(self.airsim_vz.get())
            duration = float(self.airsim_duration.get())
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
            return
        
        # Run in background thread to avoid freezing GUI
        def velocity_thread():
            try:
                self.root.after(0, lambda: self.log(f"AirSim: Setting velocity ({vx}, {vy}, {vz})...", "info"))
                # Use publish method for velocity command
                self.client.publish("/cmd_vel", "geometry_msgs/Twist", {
                    "linear": {"x": vx, "y": vy, "z": vz},
                    "duration": duration
                })
                self.root.after(0, lambda: self.log(f"AirSim: Set velocity ({vx}, {vy}, {vz}) for {duration}s", "success"))
                self.root.after(0, lambda: self.info_label.config(text=f"AirSim: Velocity set ({vx}, {vy}, {vz})"))
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: self.log(f"AirSim set_velocity error: {msg}", "error"))
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Failed to set velocity: {msg}"))
        
        threading.Thread(target=velocity_thread, daemon=True).start()
    
    def airsim_take_photo(self):
        """Take a photo with AirSim."""
        if not self.client or not self.is_connected:
            messagebox.showwarning("Warning", "Please connect to AirSim first")
            return
        if not isinstance(self.client, AirSimClient):
            messagebox.showwarning("Warning", "This action is only available for AirSim client")
            return
        
        # Get filename in main thread (file dialog must be in main thread)
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                title="Save Photo",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if not filename:
                return  # User cancelled
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get filename: {e}")
            return
        
        # Run photo capture in background thread to avoid freezing GUI
        def photo_thread():
            try:
                self.root.after(0, lambda: self.log("AirSim: Capturing photo...", "info"))
                result = self.client.take_photo(save_path=filename)
                if result.get("success"):
                    self.root.after(0, lambda: self.log(f"AirSim: Photo saved to {filename}", "success"))
                    self.root.after(0, lambda: messagebox.showinfo("Success", f"Photo saved to:\n{filename}"))
                else:
                    self.root.after(0, lambda: self.log(f"AirSim: Photo failed - {result.get('message')}", "error"))
                    self.root.after(0, lambda: messagebox.showerror("Error", result.get("message", "Failed to take photo")))
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: self.log(f"AirSim take_photo error: {msg}", "error"))
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Failed to take photo: {msg}"))
        
        threading.Thread(target=photo_thread, daemon=True).start()
    
    def airsim_get_telemetry(self):
        """Get AirSim telemetry data."""
        if not self.client or not self.is_connected:
            messagebox.showwarning("Warning", "Please connect to AirSim first")
            return
        if not isinstance(self.client, AirSimClient):
            messagebox.showwarning("Warning", "This action is only available for AirSim client")
            return
        
        # Run in background thread to avoid freezing GUI
        def telemetry_thread():
            try:
                self.root.after(0, lambda: self.log("AirSim: Retrieving telemetry...", "info"))
                telemetry = self.client.get_telemetry()
                if telemetry.get("success"):
                    # Display telemetry in a message box
                    import json
                    telemetry_str = json.dumps(telemetry, indent=2, ensure_ascii=False)
                    self.root.after(0, lambda: self.log("AirSim: Telemetry retrieved", "success"))
                    
                    # Show in a scrolled window (must be in main thread)
                    def show_telemetry():
                        from tkinter import Toplevel, scrolledtext
                        window = Toplevel(self.root)
                        window.title("AirSim Telemetry")
                        window.geometry("600x500")
                        
                        text = scrolledtext.ScrolledText(window, wrap=tk.WORD, font=('Courier', 9))
                        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                        text.insert('1.0', telemetry_str)
                        text.config(state=tk.DISABLED)
                    
                    self.root.after(0, show_telemetry)
                else:
                    self.root.after(0, lambda: self.log(f"AirSim: Telemetry failed - {telemetry.get('message')}", "error"))
                    self.root.after(0, lambda: messagebox.showerror("Error", telemetry.get("message", "Failed to get telemetry")))
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: self.log(f"AirSim get_telemetry error: {msg}", "error"))
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Failed to get telemetry: {msg}"))
        
        threading.Thread(target=telemetry_thread, daemon=True).start()


def main():
    """Main entry point."""
    import sys
    import traceback
    
    root = tk.Tk()
    app = RosClientGUITest(root)
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        """Global exception handler for unhandled exceptions."""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        print(f"Unhandled exception: {error_msg}", file=sys.stderr)
        
        # Try to log to GUI if available
        try:
            if app and hasattr(app, 'log'):
                app.root.after(0, lambda msg=str(exc_value): app.log(f"Unhandled exception: {msg}", "error"))
        except:
            pass
    
    # Set global exception handler
    sys.excepthook = handle_exception
    
    def on_closing():
        """Handle window closing."""
        app.stop_update.set()
        if app.client:
            app.disconnect()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()

