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
        self.root.title("ROS Client GUI Test Tool")
        self.root.geometry("1400x900")
        
        self.client: Optional[RosClient] = None
        self.is_connected = False
        self.update_thread: Optional[threading.Thread] = None
        self.stop_update = threading.Event()
        self.image_queue = queue.Queue(maxsize=1)
        self.point_cloud_queue = queue.Queue(maxsize=1)
        
        self.setup_ui()
        self.setup_update_loop()
        
    def setup_ui(self):
        """Setup the user interface."""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Connection tab
        self.connection_frame = ttk.Frame(notebook)
        notebook.add(self.connection_frame, text="连接配置")
        self.setup_connection_tab()
        
        # Status tab
        self.status_frame = ttk.Frame(notebook)
        notebook.add(self.status_frame, text="状态监控")
        self.setup_status_tab()
        
        # Image tab
        self.image_frame = ttk.Frame(notebook)
        notebook.add(self.image_frame, text="图像显示")
        self.setup_image_tab()
        
        # Point Cloud tab
        self.pointcloud_frame = ttk.Frame(notebook)
        notebook.add(self.pointcloud_frame, text="点云显示")
        self.setup_pointcloud_tab()
        
        # Control tab
        self.control_frame = ttk.Frame(notebook)
        notebook.add(self.control_frame, text="控制命令")
        self.setup_control_tab()
        
        # Network Test tab
        self.network_frame = ttk.Frame(notebook)
        notebook.add(self.network_frame, text="网络测试")
        self.setup_network_tab()
        
    def setup_connection_tab(self):
        """Setup connection configuration tab."""
        # Connection settings
        conn_group = ttk.LabelFrame(self.connection_frame, text="连接设置", padding=10)
        conn_group.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(conn_group, text="WebSocket地址:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.connection_url = tk.StringVar(value="ws://localhost:9090")
        ttk.Entry(conn_group, textvariable=self.connection_url, width=40).grid(
            row=0, column=1, padx=5, pady=5
        )
        
        ttk.Label(conn_group, text="连接模式:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.use_mock = tk.BooleanVar(value=False)
        ttk.Checkbutton(conn_group, text="使用Mock客户端（测试模式）", 
                       variable=self.use_mock).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # Connection buttons
        btn_frame = ttk.Frame(conn_group)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        self.connect_btn = ttk.Button(btn_frame, text="连接", command=self.connect)
        self.connect_btn.pack(side=tk.LEFT, padx=5)
        
        self.disconnect_btn = ttk.Button(btn_frame, text="断开", command=self.disconnect, state=tk.DISABLED)
        self.disconnect_btn.pack(side=tk.LEFT, padx=5)
        
        # Connection status
        status_group = ttk.LabelFrame(self.connection_frame, text="连接状态", padding=10)
        status_group.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.connection_status = scrolledtext.ScrolledText(
            status_group, height=15, width=80, wrap=tk.WORD
        )
        self.connection_status.pack(fill=tk.BOTH, expand=True)
        self.log("等待连接...")
        
    def setup_status_tab(self):
        """Setup status monitoring tab."""
        # Status display
        status_group = ttk.LabelFrame(self.status_frame, text="无人机状态", padding=10)
        status_group.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create status labels
        self.status_labels = {}
        status_fields = [
            ("连接状态", "connected"),
            ("解锁状态", "armed"),
            ("飞行模式", "mode"),
            ("电池电量", "battery"),
            ("纬度", "latitude"),
            ("经度", "longitude"),
            ("高度", "altitude"),
            ("横滚角", "roll"),
            ("俯仰角", "pitch"),
            ("偏航角", "yaw"),
            ("着陆状态", "landed"),
            ("到达目标", "reached"),
            ("返航状态", "returned"),
            ("起飞状态", "tookoff"),
        ]
        
        for i, (label, field) in enumerate(status_fields):
            row = i // 2
            col = (i % 2) * 2
            
            ttk.Label(status_group, text=f"{label}:").grid(
                row=row, column=col, sticky=tk.W, padx=5, pady=3
            )
            value_label = ttk.Label(status_group, text="N/A", foreground="gray")
            value_label.grid(row=row, column=col+1, sticky=tk.W, padx=5, pady=3)
            self.status_labels[field] = value_label
            
        # Update button
        btn_frame = ttk.Frame(status_group)
        btn_frame.grid(row=len(status_fields)//2 + 1, column=0, columnspan=4, pady=10)
        ttk.Button(btn_frame, text="刷新状态", command=self.update_status_display).pack()
        
    def setup_image_tab(self):
        """Setup image display tab."""
        if not HAS_CV2:
            ttk.Label(self.image_frame, text="需要安装 opencv-python 和 Pillow 以显示图像",
                     foreground="red").pack(pady=20)
            return
            
        # Image display
        img_group = ttk.LabelFrame(self.image_frame, text="相机图像", padding=10)
        img_group.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Image canvas
        self.image_canvas = tk.Canvas(img_group, bg="black", width=640, height=480)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Image controls
        img_controls = ttk.Frame(img_group)
        img_controls.pack(fill=tk.X, pady=5)
        
        self.auto_update_image = tk.BooleanVar(value=True)
        ttk.Checkbutton(img_controls, text="自动更新", 
                       variable=self.auto_update_image).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(img_controls, text="手动获取", 
                  command=self.fetch_image_manual).pack(side=tk.LEFT, padx=5)
        
        self.image_info_label = ttk.Label(img_controls, text="等待图像...")
        self.image_info_label.pack(side=tk.LEFT, padx=10)
        
    def setup_pointcloud_tab(self):
        """Setup point cloud display tab."""
        if not HAS_MATPLOTLIB:
            ttk.Label(self.pointcloud_frame, text="需要安装 matplotlib 和 numpy 以显示点云",
                     foreground="red").pack(pady=20)
            return
            
        # Point cloud display
        pc_group = ttk.LabelFrame(self.pointcloud_frame, text="点云数据", padding=10)
        pc_group.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Matplotlib figure
        self.pc_figure = Figure(figsize=(8, 6), dpi=100)
        self.pc_ax = self.pc_figure.add_subplot(111, projection='3d')
        self.pc_canvas = FigureCanvasTkAgg(self.pc_figure, pc_group)
        self.pc_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Point cloud controls
        pc_controls = ttk.Frame(pc_group)
        pc_controls.pack(fill=tk.X, pady=5)
        
        self.auto_update_pc = tk.BooleanVar(value=True)
        ttk.Checkbutton(pc_controls, text="自动更新", 
                       variable=self.auto_update_pc).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(pc_controls, text="手动获取", 
                  command=self.fetch_pointcloud_manual).pack(side=tk.LEFT, padx=5)
        
        self.pc_info_label = ttk.Label(pc_controls, text="等待点云数据...")
        self.pc_info_label.pack(side=tk.LEFT, padx=10)
        
    def setup_control_tab(self):
        """Setup control command tab."""
        # Topic selection
        topic_group = ttk.LabelFrame(self.control_frame, text="Topic配置", padding=10)
        topic_group.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(topic_group, text="Topic名称:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.control_topic = tk.StringVar(value="/control")
        ttk.Entry(topic_group, textvariable=self.control_topic, width=30).grid(
            row=0, column=1, padx=5, pady=5
        )
        
        ttk.Label(topic_group, text="Topic类型:").grid(row=0, column=2, sticky=tk.W, padx=10, pady=5)
        self.control_type = tk.StringVar(value="controller_msgs/cmd")
        ttk.Entry(topic_group, textvariable=self.control_type, width=30).grid(
            row=0, column=3, padx=5, pady=5
        )
        
        # Message editor
        msg_group = ttk.LabelFrame(self.control_frame, text="消息内容 (JSON格式)", padding=10)
        msg_group.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.message_editor = scrolledtext.ScrolledText(
            msg_group, height=15, width=80, wrap=tk.WORD
        )
        self.message_editor.pack(fill=tk.BOTH, expand=True)
        self.message_editor.insert("1.0", '{\n    "cmd": 1\n}')
        
        # Preset commands
        preset_group = ttk.LabelFrame(self.control_frame, text="预设命令", padding=10)
        preset_group.pack(fill=tk.X, padx=10, pady=5)
        
        presets = [
            ("起飞", '{"cmd": 1}'),
            ("降落", '{"cmd": 2}'),
            ("返航", '{"cmd": 3}'),
            ("悬停", '{"cmd": 4}'),
        ]
        
        for i, (name, cmd) in enumerate(presets):
            btn = ttk.Button(preset_group, text=name, 
                           command=lambda c=cmd: self.set_preset_command(c))
            btn.grid(row=0, column=i, padx=5)
        
        # Send button
        btn_frame = ttk.Frame(self.control_frame)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="发送命令", command=self.send_control_command,
                  style="Accent.TButton").pack()
        
        # Command history
        history_group = ttk.LabelFrame(self.control_frame, text="命令历史", padding=10)
        history_group.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.command_history = scrolledtext.ScrolledText(
            history_group, height=8, width=80, wrap=tk.WORD
        )
        self.command_history.pack(fill=tk.BOTH, expand=True)
        
    def setup_network_tab(self):
        """Setup network test tab."""
        # Test configuration
        test_group = ttk.LabelFrame(self.network_frame, text="网络测试配置", padding=10)
        test_group.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(test_group, text="测试地址:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.test_url = tk.StringVar(value="ws://localhost:9090")
        ttk.Entry(test_group, textvariable=self.test_url, width=40).grid(
            row=0, column=1, padx=5, pady=5
        )
        
        ttk.Label(test_group, text="超时时间(秒):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.test_timeout = tk.StringVar(value="5")
        ttk.Entry(test_group, textvariable=self.test_timeout, width=10).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=5
        )
        
        # Test buttons
        btn_frame = ttk.Frame(test_group)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Button(btn_frame, text="测试连接", command=self.test_connection).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="测试Topic", command=self.test_topics).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="完整测试", command=self.run_full_test).pack(side=tk.LEFT, padx=5)
        
        # Test results
        result_group = ttk.LabelFrame(self.network_frame, text="测试结果", padding=10)
        result_group.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.test_results = scrolledtext.ScrolledText(
            result_group, height=20, width=80, wrap=tk.WORD
        )
        self.test_results.pack(fill=tk.BOTH, expand=True)
        
    def setup_update_loop(self):
        """Setup periodic update loop."""
        def update_loop():
            while not self.stop_update.is_set():
                try:
                    if self.is_connected and self.client:
                        self.root.after(0, self.update_status_display)
                        if self.auto_update_image.get() and HAS_CV2:
                            self.root.after(0, self.update_image_display)
                        if self.auto_update_pc.get() and HAS_MATPLOTLIB:
                            self.root.after(0, self.update_pointcloud_display)
                except Exception as e:
                    self.log(f"Update error: {e}")
                time.sleep(1)  # Update every second
                
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
        
    def log(self, message: str):
        """Log message to connection status."""
        timestamp = time.strftime("%H:%M:%S")
        self.connection_status.insert(tk.END, f"[{timestamp}] {message}\n")
        self.connection_status.see(tk.END)
        
    def connect(self):
        """Connect to ROS bridge."""
        url = self.connection_url.get().strip()
        if not url:
            messagebox.showerror("错误", "请输入WebSocket地址")
            return
            
        try:
            self.log(f"正在连接到 {url}...")
            self.connect_btn.config(state=tk.DISABLED)
            
            if self.use_mock.get():
                self.client = MockRosClient(url)
                self.log("使用Mock客户端（测试模式）")
            else:
                self.client = RosClient(url)
                self.client.connect_async()
                
            # Wait a bit for connection
            time.sleep(2)
            
            if self.client.is_connected():
                self.is_connected = True
                self.connect_btn.config(state=tk.DISABLED)
                self.disconnect_btn.config(state=tk.NORMAL)
                self.log("连接成功！")
            else:
                self.log("连接失败，请检查地址和网络")
                messagebox.showwarning("警告", "连接失败，请检查地址和网络")
                
        except Exception as e:
            self.log(f"连接错误: {e}")
            messagebox.showerror("错误", f"连接失败: {e}")
            self.connect_btn.config(state=tk.NORMAL)
            
    def disconnect(self):
        """Disconnect from ROS bridge."""
        try:
            if self.client:
                self.client.terminate()
                self.log("已断开连接")
            self.is_connected = False
            self.client = None
            self.connect_btn.config(state=tk.NORMAL)
            self.disconnect_btn.config(state=tk.DISABLED)
        except Exception as e:
            self.log(f"断开连接错误: {e}")
            
    def update_status_display(self):
        """Update status display."""
        if not self.client or not self.is_connected:
            return
            
        try:
            state = self.client.get_status()
            pos = self.client.get_position()
            ori = self.client.get_orientation()
            
            status_data = {
                "connected": "已连接" if state.connected else "未连接",
                "armed": "已解锁" if state.armed else "已锁定",
                "mode": state.mode or "N/A",
                "battery": f"{state.battery:.1f}%",
                "latitude": f"{pos[0]:.6f}",
                "longitude": f"{pos[1]:.6f}",
                "altitude": f"{pos[2]:.2f}m",
                "roll": f"{ori[0]:.2f}°",
                "pitch": f"{ori[1]:.2f}°",
                "yaw": f"{ori[2]:.2f}°",
                "landed": "已着陆" if state.landed else "飞行中",
                "reached": "是" if state.reached else "否",
                "returned": "是" if state.returned else "否",
                "tookoff": "是" if state.tookoff else "否",
            }
            
            for field, label in self.status_labels.items():
                value = status_data.get(field, "N/A")
                label.config(text=value, foreground="black")
                
        except Exception as e:
            self.log(f"状态更新错误: {e}")
            
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
                self.image_info_label.config(
                    text=f"图像时间戳: {timestamp:.3f}, 尺寸: {frame.shape[1]}x{frame.shape[0]}"
                )
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
                frame_resized = cv2.resize(frame, (new_width, new_height))
            else:
                frame_resized = frame
                
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image=image)
            
            # Update canvas
            self.image_canvas.delete("all")
            self.image_canvas.create_image(
                canvas_width // 2, canvas_height // 2,
                image=photo, anchor=tk.CENTER
            )
            self.image_canvas.image = photo  # Keep a reference
            
        except Exception as e:
            self.log(f"图像显示错误: {e}")
            
    def fetch_image_manual(self):
        """Manually fetch image."""
        if not self.client or not self.is_connected:
            messagebox.showwarning("警告", "请先连接")
            return
            
        try:
            self.log("正在获取图像...")
            image_data = self.client.fetch_camera_image()
            if image_data:
                frame, timestamp = image_data
                self.display_image(frame)
                self.image_info_label.config(
                    text=f"图像时间戳: {timestamp:.3f}, 尺寸: {frame.shape[1]}x{frame.shape[0]}"
                )
                self.log("图像获取成功")
            else:
                self.log("未获取到图像")
                messagebox.showinfo("信息", "未获取到图像数据")
        except Exception as e:
            self.log(f"获取图像错误: {e}")
            messagebox.showerror("错误", f"获取图像失败: {e}")
            
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
                    self.pc_info_label.config(
                        text=f"点云时间戳: {timestamp:.3f}, 点数: {len(points)}"
                    )
        except Exception as e:
            pass  # Silently fail
            
    def display_pointcloud(self, points):
        """Display point cloud."""
        try:
            self.pc_ax.clear()
            
            if len(points) > 0:
                # Sample points if too many
                if len(points) > 10000:
                    indices = np.random.choice(len(points), 10000, replace=False)
                    points = points[indices]
                    
                self.pc_ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                                 c=points[:, 2], cmap='viridis', s=1)
                self.pc_ax.set_xlabel('X')
                self.pc_ax.set_ylabel('Y')
                self.pc_ax.set_zlabel('Z')
                self.pc_ax.set_title(f'Point Cloud ({len(points)} points)')
                
            self.pc_canvas.draw()
        except Exception as e:
            self.log(f"点云显示错误: {e}")
            
    def fetch_pointcloud_manual(self):
        """Manually fetch point cloud."""
        if not self.client or not self.is_connected:
            messagebox.showwarning("警告", "请先连接")
            return
            
        try:
            self.log("正在获取点云...")
            pc_data = self.client.fetch_point_cloud()
            if pc_data:
                points, timestamp = pc_data
                self.display_pointcloud(points)
                self.pc_info_label.config(
                    text=f"点云时间戳: {timestamp:.3f}, 点数: {len(points)}"
                )
                self.log(f"点云获取成功，点数: {len(points)}")
            else:
                self.log("未获取到点云")
                messagebox.showinfo("信息", "未获取到点云数据")
        except Exception as e:
            self.log(f"获取点云错误: {e}")
            messagebox.showerror("错误", f"获取点云失败: {e}")
            
    def set_preset_command(self, command: str):
        """Set preset command in editor."""
        self.message_editor.delete("1.0", tk.END)
        self.message_editor.insert("1.0", command)
        
    def send_control_command(self):
        """Send control command."""
        if not self.client or not self.is_connected:
            messagebox.showwarning("警告", "请先连接")
            return
            
        try:
            topic = self.control_topic.get().strip()
            topic_type = self.control_type.get().strip()
            message_text = self.message_editor.get("1.0", tk.END).strip()
            
            if not topic or not message_text:
                messagebox.showwarning("警告", "请填写Topic和消息内容")
                return
                
            # Parse JSON
            try:
                message = json.loads(message_text)
            except json.JSONDecodeError as e:
                messagebox.showerror("错误", f"JSON格式错误: {e}")
                return
                
            # Send command
            self.client.publish(topic, topic_type, message)
            
            # Log to history
            timestamp = time.strftime("%H:%M:%S")
            self.command_history.insert(tk.END, 
                f"[{timestamp}] {topic} ({topic_type}): {message_text}\n")
            self.command_history.see(tk.END)
            
            self.log(f"命令已发送: {topic} -> {message_text}")
            messagebox.showinfo("成功", "命令发送成功")
            
        except Exception as e:
            self.log(f"发送命令错误: {e}")
            messagebox.showerror("错误", f"发送命令失败: {e}")
            
    def test_connection(self):
        """Test ROS connection."""
        url = self.test_url.get().strip()
        timeout = float(self.test_timeout.get() or "5")
        
        self.test_results.insert(tk.END, f"\n{'='*60}\n")
        self.test_results.insert(tk.END, f"开始连接测试: {url}\n")
        self.test_results.see(tk.END)
        
        def run_test():
            try:
                test_client = RosClient(url)
                test_client.connect_async()
                
                # Wait for connection
                start_time = time.time()
                while time.time() - start_time < timeout:
                    if test_client.is_connected():
                        self.test_results.insert(tk.END, f"✓ 连接成功！\n")
                        test_client.terminate()
                        self.test_results.see(tk.END)
                        return
                    time.sleep(0.5)
                    
                self.test_results.insert(tk.END, f"✗ 连接超时\n")
                test_client.terminate()
            except Exception as e:
                self.test_results.insert(tk.END, f"✗ 连接失败: {e}\n")
            finally:
                self.test_results.see(tk.END)
                
        threading.Thread(target=run_test, daemon=True).start()
        
    def test_topics(self):
        """Test ROS topics."""
        if not self.client or not self.is_connected:
            messagebox.showwarning("警告", "请先连接到ROS")
            return
            
        self.test_results.insert(tk.END, f"\n{'='*60}\n")
        self.test_results.insert(tk.END, "开始Topic测试...\n")
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
                self.test_results.insert(tk.END, f"  - {topic}: 需要手动订阅测试\n")
            except Exception as e:
                self.test_results.insert(tk.END, f"  - {topic}: ✗ {e}\n")
                
        self.test_results.see(tk.END)
        
    def run_full_test(self):
        """Run full network test."""
        url = self.test_url.get().strip()
        timeout = float(self.test_timeout.get() or "5")
        
        self.test_results.insert(tk.END, f"\n{'='*60}\n")
        self.test_results.insert(tk.END, f"开始完整测试: {url}\n")
        self.test_results.insert(tk.END, f"{'='*60}\n")
        self.test_results.see(tk.END)
        
        def run_full():
            try:
                # Test 1: Connection
                self.test_results.insert(tk.END, "\n[1/3] 测试连接...\n")
                test_client = RosClient(url)
                test_client.connect_async()
                
                start_time = time.time()
                connected = False
                while time.time() - start_time < timeout:
                    if test_client.is_connected():
                        connected = True
                        self.test_results.insert(tk.END, "  ✓ 连接成功\n")
                        break
                    time.sleep(0.5)
                    
                if not connected:
                    self.test_results.insert(tk.END, "  ✗ 连接失败\n")
                    test_client.terminate()
                    return
                    
                # Test 2: Status
                self.test_results.insert(tk.END, "\n[2/3] 测试状态获取...\n")
                try:
                    state = test_client.get_status()
                    self.test_results.insert(tk.END, f"  ✓ 状态获取成功\n")
                    self.test_results.insert(tk.END, f"    模式: {state.mode}, 电池: {state.battery}%\n")
                except Exception as e:
                    self.test_results.insert(tk.END, f"  ✗ 状态获取失败: {e}\n")
                    
                # Test 3: Publish
                self.test_results.insert(tk.END, "\n[3/3] 测试消息发布...\n")
                try:
                    test_client.publish("/test", "std_msgs/String", {"data": "test"})
                    self.test_results.insert(tk.END, "  ✓ 消息发布成功\n")
                except Exception as e:
                    self.test_results.insert(tk.END, f"  ✗ 消息发布失败: {e}\n")
                    
                test_client.terminate()
                self.test_results.insert(tk.END, f"\n{'='*60}\n")
                self.test_results.insert(tk.END, "测试完成！\n")
                
            except Exception as e:
                self.test_results.insert(tk.END, f"\n✗ 测试错误: {e}\n")
            finally:
                self.test_results.see(tk.END)
                
        threading.Thread(target=run_full, daemon=True).start()


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

