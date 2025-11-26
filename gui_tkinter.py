#!/usr/bin/env python
"""GUI test tool for RosClient."""
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import json
import time
from typing import Optional, Dict, Any
import queue

try:
    from rosclient import RosClient, MockRosClient, DroneState, ConnectionState
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
        self.stop_update = threading.Event()
        self.image_queue = queue.Queue(maxsize=1)
        self.point_cloud_queue = queue.Queue(maxsize=1)
        
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
        
        # WebSocket address
        url_frame = ttk.Frame(conn_group)
        url_frame.pack(fill=tk.X, pady=8)
        ttk.Label(url_frame, text="WebSocket URL:").pack(side=tk.LEFT, padx=(0, 10))
        self.connection_url = tk.StringVar(value="ws://192.168.27.152:9090")
        url_entry = ttk.Entry(url_frame, textvariable=self.connection_url, width=45)
        url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Connection mode
        mode_frame = ttk.Frame(conn_group)
        mode_frame.pack(fill=tk.X, pady=8)
        self.use_mock = tk.BooleanVar(value=False)
        mock_check = ttk.Checkbutton(mode_frame, text="Use Mock Client (Test Mode)", 
                       variable=self.use_mock, command=self.on_mock_mode_changed)
        mock_check.pack(side=tk.LEFT, padx=5)
        
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
        
        quick_urls = [
            ("Local Test", "ws://localhost:9090"),
            ("Default Address", "ws://192.168.27.152:9090"),
        ]
        
        for i, (name, url) in enumerate(quick_urls):
            btn = ttk.Button(quick_frame, text=name, 
                           command=lambda u=url: self.connection_url.set(u),
                           width=12)
            btn.grid(row=i//2, column=i%2, padx=5, pady=3, sticky=tk.W+tk.E)
            quick_frame.columnconfigure(i%2, weight=1)
        
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
        
        # Preset commands
        preset_group = ttk.LabelFrame(main_pane, text="Preset Commands", padding=10)
        main_pane.add(preset_group, weight=0)
        
        presets = [
            ("Takeoff", '{"cmd": 1}'),
            ("Land", '{"cmd": 2}'),
            ("Return", '{"cmd": 3}'),
            ("Hover", '{"cmd": 4}'),
        ]
        
        for i, (name, cmd) in enumerate(presets):
            btn = ttk.Button(preset_group, text=name, 
                           command=lambda c=cmd: self.set_preset_command(c),
                           width=12)
            btn.grid(row=0, column=i, padx=5, pady=5)
        
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
        """Setup periodic update loop."""
        def update_loop():
            while not self.stop_update.is_set():
                try:
                    if self.is_connected and self.client:
                        if hasattr(self, 'auto_refresh') and self.auto_refresh.get():
                            self.root.after(0, self.update_status_display)
                        if self.auto_update_image.get() and HAS_CV2:
                            self.root.after(0, self.update_image_display)
                        if self.auto_update_pc.get() and HAS_MATPLOTLIB:
                            self.root.after(0, self.update_pointcloud_display)
                        # Update recording stats
                        if hasattr(self, 'start_record_btn') and self.start_record_btn['state'] == tk.DISABLED:
                            self.root.after(0, self.update_recording_stats)
                except Exception as e:
                    self.log(f"Update error: {e}", "error")
                time.sleep(1)  # Update every second
                
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
        
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
        if connected:
            self.status_indicator.create_oval(4, 4, 12, 12, fill='#28a745', outline='')
            self.status_indicator_text.config(text="Connected", foreground='#28a745')
            self.info_label.config(text="Connected to ROS server")
        else:
            self.status_indicator.create_oval(4, 4, 12, 12, fill='#dc3545', outline='')
            self.status_indicator_text.config(text="Disconnected", foreground='#dc3545')
            self.info_label.config(text="Disconnected")
    
    def on_mock_mode_changed(self):
        """Handle mock mode checkbox change."""
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
        """Connect to ROS bridge."""
        url = self.connection_url.get().strip()
        if not url:
            messagebox.showerror("Error", "Please enter WebSocket URL")
            return
            
        try:
            self.log(f"Connecting to {url}...", "info")
            self.connect_btn.config(state=tk.DISABLED)
            self.info_label.config(text="Connecting...")
            
            def connect_thread():
                try:
                    if self.use_mock.get():
                        # Prepare config for Mock Client
                        config = {}
                        recording_file = self.recording_file_path.get().strip()
                        if recording_file:
                            config["playback_file"] = recording_file
                            config["playback_loop"] = True  # Auto loop when file is provided
                            self.root.after(0, lambda: self.log(f"Using Mock client with recording: {recording_file}", "info"))
                        else:
                            self.root.after(0, lambda: self.log("Using Mock client (Test Mode)", "warning"))
                        
                        self.client = MockRosClient(url, config=config)
                        self.client.connect_async()
                        
                        # Enable playback controls if in playback mode
                        if recording_file and self.client.is_playback_mode():
                            self.root.after(0, lambda: self.enable_playback_controls())
                    else:
                        self.client = RosClient(url)
                        self.client.connect_async()
                        
                    # Wait a bit for connection
                    time.sleep(2)
                    
                    if self.client.is_connected():
                        self.is_connected = True
                        self.root.after(0, lambda: self.connect_btn.config(state=tk.DISABLED))
                        self.root.after(0, lambda: self.disconnect_btn.config(state=tk.NORMAL))
                        self.root.after(0, lambda: self.log("Connection successful!", "success"))
                        self.root.after(0, lambda: self.update_connection_indicator(True))
                        
                        # Enable playback controls if Mock Client
                        if isinstance(self.client, MockRosClient):
                            self.root.after(0, lambda: self.load_playback_btn.config(state=tk.NORMAL))
                            if self.client.is_playback_mode():
                                self.root.after(0, lambda: self.enable_playback_controls())
                                self.root.after(0, lambda: self.playback_file_path.set(self.client._playback_file or ""))
                    else:
                        self.root.after(0, lambda: self.log("Connection failed, please check URL and network", "error"))
                        self.root.after(0, lambda: messagebox.showwarning("Warning", "Connection failed, please check URL and network"))
                        self.root.after(0, lambda: self.connect_btn.config(state=tk.NORMAL))
                        self.root.after(0, lambda: self.info_label.config(text="Connection failed"))
                except Exception as e:
                    self.root.after(0, lambda: self.log(f"Connection error: {e}", "error"))
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Connection failed: {e}"))
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
        except Exception as e:
            self.log(f"Disconnect error: {e}", "error")
            
    def update_status_display(self):
        """Update status display."""
        if not self.client or not self.is_connected:
            return
            
        try:
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
            self.log(f"状态更新错误: {e}", "error")
            
    def update_image_display(self):
        """Update image display."""
        if not self.client or not self.is_connected:
            return
            
        try:
            # Try to get latest image
            image_data = self.client.get_latest_image()
            if image_data:
                frame, timestamp = image_data
                self.display_image(frame)
                info_text = f"Timestamp: {timestamp:.3f}, Size: {frame.shape[1]}x{frame.shape[0]}"
                self.image_info_label.config(text=info_text, foreground="#212529")
        except Exception as e:
            pass  # Silently fail to avoid spam
            
    def display_image(self, frame):
        """Display image on canvas."""
        try:
            # Resize if needed
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                scale = min(canvas_width / frame.shape[1], canvas_height / frame.shape[0])
                new_width = int(frame.shape[1] * scale)
                new_height = int(frame.shape[0] * scale)
                frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            else:
                frame_resized = frame
                
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image=image)
            
            # Update canvas (clear all including placeholder text)
            self.image_canvas.delete("all")
            self.image_canvas.create_image(
                canvas_width // 2, canvas_height // 2,
                image=photo, anchor=tk.CENTER
            )
            self.image_canvas.image = photo  # Keep a reference
            
        except Exception as e:
            self.log(f"Image display error: {e}", "error")
            
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
                self.root.after(0, lambda: self.test_results.insert(tk.END, f"Connection failed: {e}\n"))
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
                    self.root.after(0, lambda: self.test_results.insert(tk.END, f"  Status retrieval failed: {e}\n"))
                    
                # Test 3: Publish
                self.root.after(0, lambda: self.test_results.insert(tk.END, "\n[3/3] Testing message publish...\n"))
                try:
                    test_client.publish("/test", "std_msgs/String", {"data": "test"})
                    self.root.after(0, lambda: self.test_results.insert(tk.END, "  Message publish successful\n"))
                except Exception as e:
                    self.root.after(0, lambda: self.test_results.insert(tk.END, f"  Message publish failed: {e}\n"))
                    
                test_client.terminate()
                self.root.after(0, lambda: self.test_results.insert(tk.END, f"\n{'='*60}\n"))
                self.root.after(0, lambda: self.test_results.insert(tk.END, "Test completed!\n"))
                
            except Exception as e:
                self.root.after(0, lambda: self.test_results.insert(tk.END, f"\nTest error: {e}\n"))
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
                self.log("Recording started", "success")
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
                self.log("Recording stopped", "info")
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
        if not self.client or not isinstance(self.client, MockRosClient):
            messagebox.showwarning("Warning", "Please connect with Mock Client first")
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
        if not self.client or not isinstance(self.client, MockRosClient):
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
        if not self.client or not isinstance(self.client, MockRosClient):
            return
        
        try:
            success = self.client.playback_pause()
            if success:
                self.log("Playback paused", "info")
        except Exception as e:
            self.log(f"Playback pause error: {e}", "error")
    
    def playback_stop(self):
        """Stop playback."""
        if not self.client or not isinstance(self.client, MockRosClient):
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


def main():
    """Main entry point."""
    root = tk.Tk()
    app = RosClientGUITest(root)
    
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

