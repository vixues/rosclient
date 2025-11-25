#!/usr/bin/env python
"""Industrial-grade GUI for RosClient using Pygame with fighter cockpit design."""
import pygame
import threading
import json
import time
from typing import Optional, Dict, Any, List, Tuple, Callable
import queue
import math
import inspect

try:
    from rosclient import RosClient, MockRosClient, DroneState, ConnectionState
except ImportError:
    print("Warning: rosclient module not found. Please ensure it's in the Python path.")
    raise

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
    HAS_NUMPY = True
except ImportError:
    HAS_CV2 = False
    HAS_NUMPY = False
    np = None
    print("Warning: cv2/numpy not available. Image display will be disabled.")

# Point cloud rendering using simple 3D projection (no matplotlib/OpenGL required)
HAS_POINTCLOUD = True

# Open3D for advanced 3D visualization
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: open3d not available. Advanced 3D display will be disabled.")


# ============================================================================
# POINT CLOUD RENDERER - Professional Implementation
# ============================================================================

class PointCloudRenderer:
    """Professional point cloud renderer with optimized performance and features."""
    
    def __init__(self, width: int = 800, height: int = 600):
        try:
            import numpy as np
            self._np = np
        except ImportError:
            raise ImportError("numpy is required for PointCloudRenderer")
            
        self.width = width
        self.height = height
        self.camera_angle_x = 0.0
        self.camera_angle_y = 0.0
        self.zoom = 1.0
        
        # Rendering settings
        self.max_points = 20000  # Adaptive based on performance
        self.point_size = 1  # Pixels per point
        self.color_scheme = 'depth'  # 'depth', 'height', 'intensity', 'uniform'
        self.show_axes = True
        self.show_info = True
        
        # Filtering settings
        self.distance_filter_min = 0.0
        self.distance_filter_max = float('inf')
        self.use_voxel_downsample = False
        self.voxel_size = 0.1
        
        # Performance stats
        self.render_stats = {
            'total_points': 0,
            'rendered_points': 0,
            'render_time': 0.0,
            'fps': 0.0
        }
        
        # Cache
        self._last_points_hash = None
        self._last_render_params = None
        self._cached_surface = None
        
    def set_camera(self, angle_x: float, angle_y: float, zoom: float):
        """Update camera parameters."""
        self.camera_angle_x = angle_x
        self.camera_angle_y = angle_y
        self.zoom = zoom
        
    def filter_points(self, points):
        """Apply filtering to point cloud."""
        np = self._np
        if len(points) == 0:
            return points
            
        # Distance filter
        distances = np.linalg.norm(points, axis=1)
        mask = (distances >= self.distance_filter_min) & (distances <= self.distance_filter_max)
        points = points[mask]
        
        # Voxel downsampling (simple grid-based)
        if self.use_voxel_downsample and len(points) > 1000:
            points = self._voxel_downsample(points, self.voxel_size)
            
        return points
    
    def _voxel_downsample(self, points, voxel_size: float):
        """Simple voxel grid downsampling."""
        np = self._np
        # Calculate voxel indices
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        
        # Use dictionary to keep first point in each voxel
        voxel_dict = {}
        for i, idx in enumerate(voxel_indices):
            idx_tuple = tuple(idx)
            if idx_tuple not in voxel_dict:
                voxel_dict[idx_tuple] = i
        
        # Return downsampled points
        indices = list(voxel_dict.values())
        return points[indices]
    
    def adaptive_sample(self, points):
        """Adaptive sampling based on point count and zoom level."""
        np = self._np
        if len(points) <= self.max_points:
            return points
            
        # Adjust max_points based on zoom (closer = more points)
        effective_max = int(self.max_points * (1.0 + self.zoom * 0.5))
        
        if len(points) <= effective_max:
            return points
            
        # Use stratified sampling for better distribution
        # Divide into spatial bins and sample from each
        num_bins = min(100, len(points) // 1000)
        if num_bins < 2:
            # Simple uniform sampling
            step = len(points) // effective_max
            return points[::step]
        
        # Spatial binning
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        bin_size = (max_coords - min_coords) / num_bins
        
        sampled_indices = []
        points_per_bin = effective_max // num_bins
        
        for i in range(num_bins):
            for j in range(num_bins):
                for k in range(num_bins):
                    bin_min = min_coords + np.array([i, j, k]) * bin_size
                    bin_max = bin_min + bin_size
                    
                    mask = np.all((points >= bin_min) & (points < bin_max), axis=1)
                    bin_points = np.where(mask)[0]
                    
                    if len(bin_points) > 0:
                        if len(bin_points) <= points_per_bin:
                            sampled_indices.extend(bin_points)
                        else:
                            # Random sample from bin
                            selected = np.random.choice(bin_points, points_per_bin, replace=False)
                            sampled_indices.extend(selected)
        
        if len(sampled_indices) == 0:
            # Fallback to uniform sampling
            step = len(points) // effective_max
            return points[::step]
            
        return points[sampled_indices]
    
    def compute_colors(self, points, z_final, max_dist: float):
        """Compute colors based on color scheme."""
        np = self._np
        if self.color_scheme == 'uniform':
            primary_r, primary_g, primary_b = DesignSystem.COLORS['primary']
            return np.full((len(points), 3), [primary_r, primary_g, primary_b], dtype=np.uint8)
        
        # Depth-based coloring
        z_normalized = np.clip((z_final + max_dist) / (2 * max_dist), 0.0, 1.0)
        primary_r, primary_g, primary_b = DesignSystem.COLORS['primary']
        
        colors = np.zeros((len(points), 3), dtype=np.uint8)
        colors[:, 0] = (primary_r * z_normalized).astype(np.uint8)
        colors[:, 1] = (primary_g * z_normalized).astype(np.uint8)
        colors[:, 2] = primary_b
        
        if self.color_scheme == 'height':
            # Height-based coloring (Z coordinate)
            z_coords = points[:, 2]
            z_min, z_max = np.min(z_coords), np.max(z_coords)
            if z_max > z_min:
                height_normalized = (z_coords - z_min) / (z_max - z_min)
                colors[:, 0] = (primary_r * height_normalized).astype(np.uint8)
                colors[:, 1] = (primary_g * height_normalized).astype(np.uint8)
                colors[:, 2] = (primary_b * height_normalized).astype(np.uint8)
        
        return colors
    
    def render(self, points) -> Optional[pygame.Surface]:
        """Render point cloud to pygame surface with optimizations."""
        import time
        start_time = time.time()
        np = self._np
        
        if not HAS_POINTCLOUD:
            return None
            
        if points is None:
            return None
            
        if len(points) == 0:
            # Return empty surface instead of None
            surface = pygame.Surface((self.width, self.height))
            surface.fill(DesignSystem.COLORS['bg'])
            return surface
            
        try:
            # Validate input
            if not isinstance(points, np.ndarray):
                points = np.array(points, dtype=np.float32)
            
            if points.ndim != 2 or points.shape[1] < 3:
                print(f"Warning: Invalid point cloud shape: {points.shape if hasattr(points, 'shape') else type(points)}")
                # Return empty surface
                surface = pygame.Surface((self.width, self.height))
                surface.fill(DesignSystem.COLORS['bg'])
                return surface
            
            original_count = len(points)
            
            # Apply filters (but don't filter too aggressively)
            points = self.filter_points(points)
            if len(points) == 0:
                # Return empty surface instead of None
                surface = pygame.Surface((self.width, self.height))
                surface.fill(DesignSystem.COLORS['bg'])
                return surface
            
            # Adaptive sampling (preserve more points for better visualization)
            points = self.adaptive_sample(points)
            
            # Create surface
            surface = pygame.Surface((self.width, self.height))
            surface.fill(DesignSystem.COLORS['bg'])
            
            # Calculate center and scale
            center = np.mean(points, axis=0)
            points_centered = points - center
            distances = np.linalg.norm(points_centered, axis=1)
            max_dist = np.max(distances) if len(distances) > 0 else 1.0
            if max_dist == 0:
                max_dist = 1.0
            
            scale = min(self.width, self.height) * 0.4 / max_dist * self.zoom
            
            # Pre-compute rotation matrices
            cos_x, sin_x = math.cos(self.camera_angle_x), math.sin(self.camera_angle_x)
            cos_y, sin_y = math.cos(self.camera_angle_y), math.sin(self.camera_angle_y)
            
            # Vectorized rotation
            x, y, z = points_centered[:, 0], points_centered[:, 1], points_centered[:, 2]
            
            # Rotate around X axis
            y_rot = y * cos_x - z * sin_x
            z_rot = y * sin_x + z * cos_x
            
            # Rotate around Y axis
            x_final = x * cos_y + z_rot * sin_y
            z_final = -x * sin_y + z_rot * cos_y
            
            # Frustum culling (filter points in front) - less aggressive
            # Allow more points behind camera to be visible
            front_mask = z_final > -max_dist * 0.5  # Changed from 0.1 to 0.5 for more visibility
            x_final = x_final[front_mask]
            y_final = y_rot[front_mask]
            z_final = z_final[front_mask]
            
            if len(x_final) == 0:
                # Still return surface even if no points visible
                return surface
            
            # Perspective projection (vectorized)
            z_scale = 1.0 + z_final / max_dist
            proj_x_float = self.width / 2 + x_final * scale / z_scale
            proj_y_float = self.height / 2 - y_final * scale / z_scale
            
            # Convert to integer coordinates and clip to bounds FIRST
            # This ensures all coordinates are always within valid range
            proj_x = np.clip(proj_x_float.astype(np.int32), 0, self.width - 1)
            proj_y = np.clip(proj_y_float.astype(np.int32), 0, self.height - 1)
            
            # All points should be valid after clip, but verify
            # Note: After clip, all values are in [0, width-1] and [0, height-1]
            # So we can use all points directly
            
            # Get the corresponding original points for color computation
            # front_mask was applied to x_final, y_final, z_final
            # We need the corresponding points from the sampled/filtered points
            # Since we already filtered and sampled, we use the current points array
            # and select based on front_mask
            filtered_points = points[front_mask] if len(points) == len(front_mask) else points
            
            # Compute colors for all points (they're all valid after clip)
            colors = self.compute_colors(filtered_points, z_final, max_dist)
            
            # Vectorized point drawing using pixel array
            # Note: pygame.surfarray.pixels3d uses [y, x] indexing
            pixel_array = pygame.surfarray.pixels3d(surface)
            
            # Draw points (vectorized for better performance)
            # All coordinates are guaranteed to be in valid range after clip
            if len(proj_x) > 0 and len(proj_y) > 0:
                try:
                    if self.point_size == 1:
                        # Single pixel - fastest method
                        # Direct vectorized assignment with guaranteed valid indices
                        # pixel_array shape is (height, width, 3)
                        # We use [y, x] indexing where y is row (0 to height-1), x is col (0 to width-1)
                        # Final safety: ensure arrays are same length
                        min_len = min(len(proj_x), len(proj_y), len(colors))
                        if min_len > 0:
                            pixel_array[proj_y[:min_len], proj_x[:min_len]] = colors[:min_len]
                    else:
                        # Multi-pixel points with bounds checking
                        for i in range(len(proj_x)):
                            px, py = int(proj_x[i]), int(proj_y[i])
                            # Double-check bounds (shouldn't be needed after clip, but safety)
                            if 0 <= px < self.width and 0 <= py < self.height:
                                color = colors[i]
                                size = self.point_size
                                half_size = size // 2
                                # Draw square of pixels
                                for dy in range(-half_size, half_size + 1):
                                    for dx in range(-half_size, half_size + 1):
                                        x_pos, y_pos = px + dx, py + dy
                                        if 0 <= x_pos < self.width and 0 <= y_pos < self.height:
                                            pixel_array[y_pos, x_pos] = color
                except (IndexError, ValueError) as e:
                    # Fallback: draw points one by one if vectorized method fails
                    print(f"Vectorized drawing failed, using fallback: {e}")
                    for i in range(min(len(proj_x), len(proj_y), len(colors))):
                        px, py = int(proj_x[i]), int(proj_y[i])
                        if 0 <= px < self.width and 0 <= py < self.height:
                            try:
                                pixel_array[py, px] = colors[i]
                            except (IndexError, ValueError):
                                continue
            
            del pixel_array  # Unlock surface
            
            # Draw axes with fixed world coordinate length (relative to point cloud scale)
            if self.show_axes:
                # Use fixed world coordinate length based on point cloud scale
                # This ensures axes maintain consistent size relative to the point cloud
                # Axis length is a fraction of max_dist to keep it proportional
                axis_world_length = max_dist * 0.2  # 20% of point cloud extent
                
                # Origin is at point cloud center (which is at screen center after centering)
                origin_x, origin_y = self.width // 2, self.height // 2
                
                # World coordinate axes (relative to point cloud center)
                # These are in the same coordinate system as the point cloud
                axis_points = np.array([
                    [axis_world_length, 0, 0],  # X axis (red)
                    [0, axis_world_length, 0],  # Y axis (green)
                    [0, 0, axis_world_length],  # Z axis (blue)
                ], dtype=np.float32)
                
                axis_colors = [
                    DesignSystem.COLORS['error'],    # X - Red
                    DesignSystem.COLORS['success'],  # Y - Green
                    DesignSystem.COLORS['primary'],  # Z - Blue/White
                ]
                
                # Project axes using the same center and scale as point cloud
                # This ensures axes are correctly positioned and scaled
                for axis_point, color in zip(axis_points, axis_colors):
                    # Project axis endpoint using same transformation as points
                    axis_centered = axis_point  # Already relative to center
                    
                    # Apply same rotation as point cloud
                    x, y, z = axis_centered[0], axis_centered[1], axis_centered[2]
                    y_rot = y * cos_x - z * sin_x
                    z_rot = y * sin_x + z * cos_x
                    x_final = x * cos_y + z_rot * sin_y
                    z_final = -x * sin_y + z_rot * cos_y
                    
                    # Apply same perspective projection as points
                    z_scale = 1.0 + z_final / max_dist
                    axis_proj_x = int(self.width / 2 + x_final * scale / z_scale)
                    axis_proj_y = int(self.height / 2 - y_rot * scale / z_scale)
                    
                    # Only draw if axis endpoint is visible
                    if 0 <= axis_proj_x < self.width and 0 <= axis_proj_y < self.height:
                        # Draw axis line from origin to endpoint
                        pygame.draw.line(surface, color, 
                                       (origin_x, origin_y), 
                                       (axis_proj_x, axis_proj_y), 3)
                        
                        # Draw axis label at endpoint
                        font = DesignSystem.get_font('small')
                        axis_labels = ['X', 'Y', 'Z']
                        label_idx = list(axis_colors).index(color)
                        if label_idx < len(axis_labels):
                            label_surf = font.render(axis_labels[label_idx], True, color)
                            # Position label slightly offset from endpoint
                            label_pos = (axis_proj_x + 5, axis_proj_y - 5)
                            surface.blit(label_surf, label_pos)
            
            # Draw info text
            if self.show_info:
                font = DesignSystem.get_font('small')
                render_time = (time.time() - start_time) * 1000
                info_text = (f"Points: {original_count} → {len(proj_x)} | "
                           f"Zoom: {self.zoom:.2f} | "
                           f"Time: {render_time:.1f}ms")
                text_surf = font.render(info_text, True, DesignSystem.COLORS['text_secondary'])
                surface.blit(text_surf, (10, 10))
            
            # Update stats
            render_time = time.time() - start_time
            self.render_stats = {
                'total_points': original_count,
                'rendered_points': len(proj_x) if 'proj_x' in locals() else 0,
                'render_time': render_time,
                'fps': 1.0 / render_time if render_time > 0 else 0
            }
            
            return surface
            
        except Exception as e:
            print(f"Point cloud render error: {e}")
            import traceback
            traceback.print_exc()
            # Return empty surface instead of None to prevent display issues
            try:
                surface = pygame.Surface((self.width, self.height))
                surface.fill(DesignSystem.COLORS['bg'])
                return surface
            except:
                return None
    
    def _project_point_3d(self, point, center, max_dist: float, scale: float) -> Optional[Tuple[int, int]]:
        """Project a 3D point to 2D screen coordinates."""
        try:
            # Use point directly (center is already accounted for in caller if needed)
            # For fixed axes, center should be [0,0,0]
            x, y, z = point[0] - center[0], point[1] - center[1], point[2] - center[2]
            
            cos_x, sin_x = math.cos(self.camera_angle_x), math.sin(self.camera_angle_x)
            cos_y, sin_y = math.cos(self.camera_angle_y), math.sin(self.camera_angle_y)
            
            # Rotate around X axis
            y_rot = y * cos_x - z * sin_x
            z_rot = y * sin_x + z * cos_x
            
            # Rotate around Y axis
            x_final = x * cos_y + z_rot * sin_y
            z_final = -x * sin_y + z_rot * cos_y
            
            # Project to 2D (allow points behind camera for axes)
            if z_final > -max_dist * 2.0:  # More lenient for axes
                if abs(z_final) > 0.001:
                    proj_x = int(self.width / 2 + x_final * scale / (1 + z_final / max_dist))
                    proj_y = int(self.height / 2 - y_rot * scale / (1 + z_final / max_dist))
                else:
                    proj_x = int(self.width / 2 + x_final * scale)
                    proj_y = int(self.height / 2 - y_rot * scale)
                
                if 0 <= proj_x < self.width and 0 <= proj_y < self.height:
                    return (proj_x, proj_y)
        except:
            pass
        return None


# ============================================================================
# DESIGN SYSTEM: Fighter Cockpit + Console Style
# ============================================================================

class DesignSystem:
    """Industrial fighter cockpit design system with console typography."""
    
    # Black Color Palette - Industrial black theme
    COLORS = {
        # Background layers - Pure black tones
        'bg': (5, 5, 5),              # Pure black
        'bg_secondary': (10, 10, 10),  # Slightly lighter black
        'bg_panel': (15, 15, 15),      # Panel background
        
        # Surface layers - Dark grays
        'surface': (20, 20, 20),       # Main surface
        'surface_light': (30, 30, 30), # Light surface
        'surface_hover': (40, 40, 40),  # Hover state
        'surface_active': (50, 50, 50), # Active state
        
        # Primary colors - White/Gray (minimal, industrial)
        'primary': (255, 255, 255),     # Pure white
        'primary_dark': (200, 200, 200), # Dark gray
        'primary_light': (255, 255, 255), # White
        'primary_glow': (255, 255, 255, 60), # Glow effect
        
        # Status colors - Subtle, high contrast
        'success': (0, 255, 0),         # Green (operational)
        'success_glow': (0, 255, 0, 40),
        'warning': (255, 200, 0),       # Amber (caution)
        'warning_glow': (255, 200, 0, 40),
        'error': (255, 0, 0),            # Red (critical)
        'error_glow': (255, 0, 0, 40),
        
        # Text colors - White/Gray scale
        'text': (255, 255, 255),        # Pure white
        'text_secondary': (180, 180, 180), # Light gray
        'text_tertiary': (120, 120, 120),   # Medium gray
        'text_console': (0, 255, 0),        # Green console
        'text_label': (200, 200, 200),      # Light gray label
        
        # Border and accent - Gray scale
        'border': (60, 60, 60),         # Medium gray border
        'border_light': (100, 100, 100), # Light gray border
        'border_glow': (255, 255, 255, 80), # White glowing border
        'accent': (150, 150, 150),       # Gray accent
        'accent_glow': (150, 150, 150, 40),
        
        # Shadow and overlay
        'shadow': (0, 0, 0, 200),       # Strong black shadow
        'shadow_light': (0, 0, 0, 120),  # Light black shadow
        'overlay': (0, 0, 0, 150),      # Black overlay
    }
    
    # Typography - Console monospace style
    FONTS = {
        'console': None,  # Will be initialized with monospace
        'label': None,     # Medium size
        'title': None,     # Large size
        'small': None,     # Small size
    }
    
    # Spacing system
    SPACING = {
        'xs': 4,
        'sm': 8,
        'md': 12,
        'lg': 16,
        'xl': 24,
        'xxl': 32,
    }
    
    # Border radius
    RADIUS = {
        'none': 0,
        'sm': 4,
        'md': 6,
        'lg': 8,
        'xl': 12,
    }
    
    # Shadow offsets
    SHADOW_OFFSET = 3
    
    @staticmethod
    def init_fonts():
        """Initialize fonts with console monospace style."""
        # Try to load monospace font, fallback to default
        try:
            # Try common monospace fonts
            font_paths = [
                'consola.ttf', 'consolas.ttf', 'Courier New.ttf',
                'DejaVuSansMono.ttf', 'LiberationMono-Regular.ttf'
            ]
            font_found = None
            for path in font_paths:
                try:
                    font_found = pygame.font.Font(path, 14)
                    break
                except:
                    continue
            
            if font_found:
                DesignSystem.FONTS['console'] = font_found
                DesignSystem.FONTS['label'] = pygame.font.Font(font_paths[0] if font_found else None, 16)
                DesignSystem.FONTS['title'] = pygame.font.Font(font_paths[0] if font_found else None, 24)
                DesignSystem.FONTS['small'] = pygame.font.Font(font_paths[0] if font_found else None, 12)
            else:
                # Fallback to default monospace
                DesignSystem.FONTS['console'] = pygame.font.Font(pygame.font.get_default_font(), 14)
                DesignSystem.FONTS['label'] = pygame.font.Font(pygame.font.get_default_font(), 16)
                DesignSystem.FONTS['title'] = pygame.font.Font(pygame.font.get_default_font(), 24)
                DesignSystem.FONTS['small'] = pygame.font.Font(pygame.font.get_default_font(), 12)
        except:
            # Ultimate fallback
            DesignSystem.FONTS['console'] = pygame.font.Font(None, 14)
            DesignSystem.FONTS['label'] = pygame.font.Font(None, 16)
            DesignSystem.FONTS['title'] = pygame.font.Font(None, 24)
            DesignSystem.FONTS['small'] = pygame.font.Font(None, 12)
    
    @staticmethod
    def get_font(size='label'):
        """Get font by size name."""
        return DesignSystem.FONTS.get(size, DesignSystem.FONTS['label'])


# ============================================================================
# BASE UI COMPONENTS
# ============================================================================

class ComponentPort:
    """Port system for component communication and message passing."""
    
    def __init__(self, name: str, port_type: str = 'signal'):
        """
        Initialize a component port.
        
        Args:
            name: Port name/identifier
            port_type: Type of port - 'signal' (control signals), 
                      'callback' (callback functions), 'param' (parameters)
        """
        self.name = name
        self.port_type = port_type
        self.connections: List[Callable] = []
        self.value = None
        self.last_value = None
        
    def connect(self, handler: Callable):
        """Connect a handler function to this port."""
        if handler not in self.connections:
            self.connections.append(handler)
    
    def disconnect(self, handler: Callable):
        """Disconnect a handler from this port."""
        if handler in self.connections:
            self.connections.remove(handler)
    
    def emit(self, value: Any = None):
        """Emit a signal/value through this port."""
        self.value = value
        for handler in self.connections:
            try:
                if self.port_type == 'callback':
                    # Check if handler accepts parameters
                    try:
                        sig = inspect.signature(handler)
                        # Get parameters excluding 'self' for bound methods
                        params = list(sig.parameters.values())
                        # Remove 'self' if it's a bound method
                        if hasattr(handler, '__self__') and params and params[0].name == 'self':
                            params = params[1:]
                        
                        # Check if handler accepts any parameters (excluding self)
                        if len(params) == 0:
                            # Handler doesn't accept parameters
                            handler()
                        else:
                            # Handler accepts parameters, pass the value
                            handler(value)
                    except (ValueError, TypeError):
                        # If signature inspection fails, try calling with value
                        # and fall back to no args if it fails
                        try:
                            handler(value)
                        except TypeError:
                            handler()
                elif self.port_type == 'signal':
                    handler()
                elif self.port_type == 'param':
                    handler(self.name, value)
            except Exception as e:
                print(f"Error in port {self.name} handler: {e}")
    
    def get(self) -> Any:
        """Get current port value."""
        return self.value


class UIComponent:
    """Base class for all UI components with port-based message passing."""
    
    def __init__(self, x: int, y: int, width: int, height: int):
        self.rect = pygame.Rect(x, y, width, height)
        self.visible = True
        self.enabled = True
        
        # Port system for message passing
        self.ports: Dict[str, ComponentPort] = {}
        self._setup_ports()
        
    def _setup_ports(self):
        """Setup default ports. Override in subclasses for custom ports."""
        # Control signal ports
        self.add_port('click', 'signal')
        self.add_port('hover', 'signal')
        self.add_port('focus', 'signal')
        self.add_port('change', 'signal')
        
        # Callback ports
        self.add_port('on_click', 'callback')
        self.add_port('on_hover', 'callback')
        self.add_port('on_change', 'callback')
        
        # Parameter ports
        self.add_port('value', 'param')
        self.add_port('config', 'param')
        
    def add_port(self, name: str, port_type: str = 'signal') -> ComponentPort:
        """Add a new port to this component."""
        port = ComponentPort(name, port_type)
        self.ports[name] = port
        return port
    
    def get_port(self, name: str) -> Optional[ComponentPort]:
        """Get a port by name."""
        return self.ports.get(name)
    
    def connect_port(self, port_name: str, handler: Callable):
        """Connect a handler to a port."""
        port = self.get_port(port_name)
        if port:
            port.connect(handler)
        else:
            # Auto-create port if it doesn't exist
            port = self.add_port(port_name, 'callback')
            port.connect(handler)
    
    def emit_signal(self, port_name: str, value: Any = None):
        """Emit a signal through a port."""
        port = self.get_port(port_name)
        if port:
            port.emit(value)
        
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle pygame event. Returns True if event was handled."""
        return False
        
    def update(self, dt: float):
        """Update component state."""
        pass
        
    def draw(self, surface: pygame.Surface):
        """Draw component to surface."""
        pass


class Panel(UIComponent):
    """Panel container component with fighter cockpit styling."""
    
    def __init__(self, x: int, y: int, width: int, height: int, 
                 title: str = "", show_border: bool = True):
        super().__init__(x, y, width, height)
        self.title = title
        self.show_border = show_border
        self.children: List[UIComponent] = []
        
    def add_child(self, child: UIComponent):
        """Add child component."""
        self.children.append(child)
        
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle events for panel and children."""
        if not self.visible or not self.enabled:
            return False
            
        # Transform event position relative to panel
        if event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            if self.rect.collidepoint(event.pos):
                # Create relative event
                rel_event = pygame.event.Event(event.type)
                rel_event.pos = (event.pos[0] - self.rect.x, event.pos[1] - self.rect.y)
                rel_event.button = getattr(event, 'button', None)
                
                # Pass to children
                for child in reversed(self.children):  # Reverse for z-order
                    if child.handle_event(rel_event):
                        return True
        return False
        
    def update(self, dt: float):
        """Update panel and children."""
        for child in self.children:
            child.update(dt)
            
    def draw(self, surface: pygame.Surface):
        """Draw panel with fighter cockpit style."""
        if not self.visible:
            return
            
        # Draw panel background
        pygame.draw.rect(surface, DesignSystem.COLORS['bg_panel'], self.rect, 
                        border_radius=DesignSystem.RADIUS['md'])
        
        # Draw border with glow effect
        if self.show_border:
            # Outer border
            pygame.draw.rect(surface, DesignSystem.COLORS['border'], self.rect, 
                           width=1, border_radius=DesignSystem.RADIUS['md'])
            # Inner glow line
            inner_rect = self.rect.inflate(-2, -2)
            pygame.draw.rect(surface, DesignSystem.COLORS['border_light'], inner_rect, 
                           width=1, border_radius=DesignSystem.RADIUS['sm'])
        
        # Draw title if present
        if self.title:
            font = DesignSystem.get_font('label')
            title_surf = font.render(self.title, True, DesignSystem.COLORS['text_label'])
            title_rect = title_surf.get_rect(topleft=(self.rect.x + DesignSystem.SPACING['md'], 
                                                      self.rect.y + DesignSystem.SPACING['sm']))
            surface.blit(title_surf, title_rect)
            
            # Draw title underline
            underline_y = title_rect.bottom + 2
            pygame.draw.line(surface, DesignSystem.COLORS['primary'], 
                           (title_rect.left, underline_y),
                           (title_rect.right, underline_y), 1)
        
        # Draw children
        for child in self.children:
            # Create subsurface for clipping
            child_surface = surface.subsurface(
                pygame.Rect(child.rect.x + self.rect.x, 
                           child.rect.y + self.rect.y,
                           child.rect.width, child.rect.height)
            )
            child.draw(child_surface)


class Card(UIComponent):
    """Card component with shadow and border."""
    
    def __init__(self, x: int, y: int, width: int, height: int,
                 title: str = "", show_shadow: bool = True):
        super().__init__(x, y, width, height)
        self.title = title
        self.show_shadow = show_shadow
        self.children: List[UIComponent] = []
        
    def add_child(self, child: UIComponent):
        """Add child component."""
        self.children.append(child)
        
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle events for card and children."""
        if not self.visible or not self.enabled:
            return False
            
        if event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            if hasattr(event, 'pos') and self.rect.collidepoint(event.pos):
                # Get content area (below title)
                header_height = 36 if self.title else 0
                content_area = pygame.Rect(
                    self.rect.x,
                    self.rect.y + header_height,
                    self.rect.width,
                    self.rect.height - header_height
                )
                
                # Convert event position to content area coordinates (relative to content area start)
                rel_event = pygame.event.Event(event.type)
                rel_event.pos = (event.pos[0] - content_area.x, event.pos[1] - content_area.y)
                rel_event.button = getattr(event, 'button', None)
                rel_event.buttons = getattr(event, 'buttons', None)
                rel_event.rel = getattr(event, 'rel', None)
                
                for child in reversed(self.children):
                    if child.handle_event(rel_event):
                        return True
        return False
        
    def update(self, dt: float):
        """Update card and children."""
        for child in self.children:
            child.update(dt)
            
    def draw(self, surface: pygame.Surface):
        """Draw card with fighter cockpit styling and proper children space control."""
        if not self.visible:
            return
            
        # Draw shadow
        if self.show_shadow:
            shadow_rect = self.rect.copy()
            shadow_rect.x += DesignSystem.SHADOW_OFFSET
            shadow_rect.y += DesignSystem.SHADOW_OFFSET
            shadow_surf = pygame.Surface((shadow_rect.width, shadow_rect.height), pygame.SRCALPHA)
            shadow_color = (*DesignSystem.COLORS['shadow'][:3], 
                          DesignSystem.COLORS['shadow'][3] if len(DesignSystem.COLORS['shadow']) > 3 
                          else DesignSystem.COLORS['shadow'][0])
            pygame.draw.rect(shadow_surf, shadow_color, shadow_surf.get_rect(), 
                           border_radius=DesignSystem.RADIUS['lg'])
            surface.blit(shadow_surf, shadow_rect)
        
        # Draw card background
        pygame.draw.rect(surface, DesignSystem.COLORS['surface'], self.rect,
                        border_radius=DesignSystem.RADIUS['lg'])
        
        # Draw border with subtle glow
        pygame.draw.rect(surface, DesignSystem.COLORS['border'], self.rect,
                        width=1, border_radius=DesignSystem.RADIUS['lg'])
        
        # Calculate title header height
        header_height = 36 if self.title else 0
        
        # Draw title header if present
        if self.title:
            header_rect = pygame.Rect(self.rect.x, self.rect.y, self.rect.width, header_height)
            pygame.draw.rect(surface, DesignSystem.COLORS['surface_light'], header_rect,
                           border_radius=DesignSystem.RADIUS['lg'])
            pygame.draw.rect(surface, DesignSystem.COLORS['border_light'], header_rect,
                           width=1, border_radius=DesignSystem.RADIUS['lg'])
            
            font = DesignSystem.get_font('label')
            title_surf = font.render(self.title, True, DesignSystem.COLORS['text'])
            title_y = header_rect.y + (header_rect.height - title_surf.get_height()) // 2
            surface.blit(title_surf, (header_rect.x + DesignSystem.SPACING['md'], title_y))
        
        # Calculate content area (below title) - absolute coordinates for clipping
        content_area_abs = pygame.Rect(
            self.rect.x,
            self.rect.y + header_height,
            self.rect.width,
            self.rect.height - header_height
        )
        
        # Draw children with clipping to prevent covering title
        old_clip = surface.get_clip()
        # Set clip region to content area (below title) - use absolute coordinates
        surface.set_clip(content_area_abs)
        
        for child in self.children:
            # Child coordinates are relative to card's content area
            # Calculate absolute position for drawing
            child_abs_rect = pygame.Rect(
                self.rect.x + child.rect.x,
                self.rect.y + header_height + child.rect.y,
                child.rect.width,
                child.rect.height
            )
            
            # Only draw if child intersects with content area
            if child_abs_rect.colliderect(content_area_abs):
                # Create subsurface for child (relative to content area)
                child_surface = surface.subsurface(child_abs_rect)
                # Save child's original position temporarily
                orig_x, orig_y = child.rect.x, child.rect.y
                # Set child position to 0,0 relative to its surface
                child.rect.x = 0
                child.rect.y = 0
                child.draw(child_surface)
                # Restore original position
                child.rect.x, child.rect.y = orig_x, orig_y
        
        # Restore original clip
        surface.set_clip(old_clip)
    
    def get_content_area(self) -> pygame.Rect:
        """Get the content area rect (below title) for children placement.
        Returns rect with coordinates relative to card's position (0,0 at content area start)."""
        header_height = 36 if self.title else 0
        return pygame.Rect(
            0,  # Relative to card's content area start
            0,  # Relative to card's content area start
            self.rect.width,
            self.rect.height - header_height
        )


class Label(UIComponent):
    """Text label component."""
    
    def __init__(self, x: int, y: int, text: str, 
                 font_size: str = 'label', color: Tuple[int, int, int] = None,
                 align: str = 'left'):
        super().__init__(x, y, 0, 0)
        self.text = text
        self.font_size = font_size
        self.color = color or DesignSystem.COLORS['text']
        self.align = align  # 'left', 'center', 'right'
        self._update_size()
        
    def _update_size(self):
        """Update label size based on text."""
        font = DesignSystem.get_font(self.font_size)
        text_surf = font.render(self.text, True, self.color)
        self.rect.width = text_surf.get_width()
        self.rect.height = text_surf.get_height()
        
    def set_text(self, text: str):
        """Update label text."""
        self.text = text
        self._update_size()
        
    def draw(self, surface: pygame.Surface):
        """Draw label."""
        if not self.visible:
            return
            
        font = DesignSystem.get_font(self.font_size)
        text_surf = font.render(self.text, True, self.color)
        
        if self.align == 'center':
            pos = (self.rect.centerx - text_surf.get_width() // 2, self.rect.y)
        elif self.align == 'right':
            pos = (self.rect.right - text_surf.get_width(), self.rect.y)
        else:
            pos = (self.rect.x, self.rect.y)
            
        surface.blit(text_surf, pos)


class Field(UIComponent):
    """Field component (label + value display)."""
    
    def __init__(self, x: int, y: int, width: int, height: int,
                 label: str, value: str = "", value_color: Tuple[int, int, int] = None):
        super().__init__(x, y, width, height)
        self.label = label
        self.value = value
        self.value_color = value_color or DesignSystem.COLORS['text']
        
    def set_value(self, value: str, color: Tuple[int, int, int] = None):
        """Update field value."""
        self.value = value
        if color:
            self.value_color = color
            
    def draw(self, surface: pygame.Surface):
        """Draw field."""
        if not self.visible:
            return
            
        # Draw background
        pygame.draw.rect(surface, DesignSystem.COLORS['surface_light'], self.rect,
                       border_radius=DesignSystem.RADIUS['sm'])
        pygame.draw.rect(surface, DesignSystem.COLORS['border'], self.rect,
                       width=1, border_radius=DesignSystem.RADIUS['sm'])
        
        font = DesignSystem.get_font('label')
        
        # Draw label
        label_surf = font.render(f"{self.label}:", True, DesignSystem.COLORS['text_label'])
        label_y = self.rect.y + (self.rect.height - label_surf.get_height()) // 2
        surface.blit(label_surf, (self.rect.x + DesignSystem.SPACING['md'], label_y))
        
        # Draw value
        value_surf = font.render(str(self.value), True, self.value_color)
        value_x = self.rect.right - value_surf.get_width() - DesignSystem.SPACING['md']
        value_y = self.rect.y + (self.rect.height - value_surf.get_height()) // 2
        surface.blit(value_surf, (value_x, value_y))


# ============================================================================
# INTERACTIVE COMPONENTS
# ============================================================================

class Button(UIComponent):
    """Button component with fighter cockpit styling."""
    
    def __init__(self, x: int, y: int, width: int, height: int, text: str,
                 callback: Callable = None, color: Tuple[int, int, int] = None):
        super().__init__(x, y, width, height)
        self.text = text
        self.color = color or DesignSystem.COLORS['primary']
        self.hovered = False
        self.pressed = False
        self.animation_scale = 1.0
        
        # Connect callback to port system if provided
        if callback:
            self.connect_port('on_click', callback)
        
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle button events with port system integration."""
        if not self.visible or not self.enabled:
            return False
            
        if event.type == pygame.MOUSEMOTION:
            was_hovered = self.hovered
            self.hovered = self.rect.collidepoint(event.pos) if hasattr(event, 'pos') else False
            if was_hovered != self.hovered:
                self.emit_signal('hover', self.hovered)
                self.emit_signal('on_hover', {'hovered': self.hovered, 'text': self.text})
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if hasattr(event, 'pos') and self.rect.collidepoint(event.pos):
                self.pressed = True
                self.emit_signal('click', {'action': 'press', 'text': self.text})
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            if self.pressed:
                self.pressed = False
                if hasattr(event, 'pos') and self.rect.collidepoint(event.pos):
                    # Emit signals through port system
                    self.emit_signal('click', {'action': 'release', 'text': self.text})
                    # Get on_click port and call handlers directly to avoid parameter issues
                    on_click_port = self.get_port('on_click')
                    if on_click_port:
                        for handler in on_click_port.connections:
                            try:
                                # Check if handler is a lambda with default arguments (captured values)
                                # Lambda functions with captured values should be called without arguments
                                is_lambda = (hasattr(handler, '__name__') and 
                                           (handler.__name__ == '<lambda>' or 
                                            handler.__name__ == '<function>'))
                                
                                # Check if handler accepts parameters
                                try:
                                    sig = inspect.signature(handler)
                                    params = list(sig.parameters.values())
                                    if hasattr(handler, '__self__') and params and params[0].name == 'self':
                                        params = params[1:]
                                    
                                    # For lambda functions, check if all params have default values
                                    if is_lambda and params:
                                        all_have_defaults = all(p.default != inspect.Parameter.empty for p in params)
                                        if all_have_defaults:
                                            # Lambda with captured values - call without args
                                            handler()
                                        elif len(params) == 0:
                                            handler()
                                        else:
                                            # Try with dict, fallback to no args
                                            try:
                                                handler({'text': self.text, 'button': self})
                                            except TypeError:
                                                handler()
                                    elif len(params) == 0:
                                        handler()
                                    else:
                                        # Try with dict first, fallback to no args
                                        try:
                                            handler({'text': self.text, 'button': self})
                                        except TypeError:
                                            handler()
                                except (ValueError, TypeError):
                                    # If signature inspection fails, try calling without args first
                                    # (for lambda functions with captured values)
                                    try:
                                        handler()
                                    except TypeError:
                                        try:
                                            handler({'text': self.text, 'button': self})
                                        except TypeError:
                                            handler()
                            except Exception as e:
                                print(f"Error in button on_click handler: {e}")
                return True
        return False
        
    def update(self, dt: float):
        """Update button animation."""
        target_scale = 1.05 if self.hovered else 1.0
        if abs(self.animation_scale - target_scale) > 0.01:
            diff = target_scale - self.animation_scale
            self.animation_scale += diff * dt * 10
            
    def draw(self, surface: pygame.Surface):
        """Draw button with fighter cockpit style."""
        if not self.visible:
            return
            
        # Calculate color based on state
        if self.pressed:
            bg_color = tuple(max(0, c - 40) for c in self.color)
        elif self.hovered:
            bg_color = tuple(min(255, int(c * 1.2)) for c in self.color)
        else:
            bg_color = self.color
            
        # Draw button with scale animation
        scale = self.animation_scale
        scaled_rect = pygame.Rect(
            self.rect.centerx - self.rect.width * scale / 2,
            self.rect.centery - self.rect.height * scale / 2,
            self.rect.width * scale,
            self.rect.height * scale
        )
        
        # Draw shadow
        shadow_rect = scaled_rect.copy()
        shadow_rect.x += 2
        shadow_rect.y += 2
        shadow_surf = pygame.Surface((shadow_rect.width, shadow_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surf, (*DesignSystem.COLORS['shadow_light'][:3], 100),
                        shadow_surf.get_rect(), border_radius=DesignSystem.RADIUS['md'])
        surface.blit(shadow_surf, shadow_rect)
        
        # Draw button
        pygame.draw.rect(surface, bg_color, scaled_rect,
                        border_radius=DesignSystem.RADIUS['md'])
        pygame.draw.rect(surface, DesignSystem.COLORS['border_light'], scaled_rect,
                        width=1, border_radius=DesignSystem.RADIUS['md'])
        
        # Draw text - ensure contrast with background
        font = DesignSystem.get_font('label')
        # Calculate text color based on background brightness
        bg_brightness = sum(bg_color) / 3.0
        if bg_brightness > 200:  # Light background - use dark text
            text_color = (0, 0, 0)  # Black
        else:  # Dark background - use light text
            text_color = DesignSystem.COLORS['text']
        text_surf = font.render(self.text, True, text_color)
        text_rect = text_surf.get_rect(center=scaled_rect.center)
        surface.blit(text_surf, text_rect)


class TextInput(UIComponent):
    """Text input component with console style."""
    
    def __init__(self, x: int, y: int, width: int, height: int,
                 default_text: str = "", placeholder: str = ""):
        super().__init__(x, y, width, height)
        self.text = default_text
        self.placeholder = placeholder
        self.active = False
        self.cursor_visible = True
        self.cursor_timer = 0.0
        self.cursor_pos = len(self.text)
        
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle text input events with port system integration."""
        if not self.visible or not self.enabled:
            return False
            
        if event.type == pygame.MOUSEBUTTONDOWN:
            was_active = self.active
            self.active = self.rect.collidepoint(event.pos) if hasattr(event, 'pos') else False
            if was_active != self.active:
                self.emit_signal('focus', self.active)
            return self.active
        elif event.type == pygame.KEYDOWN and self.active:
            old_text = self.text
            if event.key == pygame.K_BACKSPACE:
                if self.cursor_pos > 0:
                    self.text = self.text[:self.cursor_pos-1] + self.text[self.cursor_pos:]
                    self.cursor_pos = max(0, self.cursor_pos - 1)
            elif event.key == pygame.K_DELETE:
                if self.cursor_pos < len(self.text):
                    self.text = self.text[:self.cursor_pos] + self.text[self.cursor_pos+1:]
            elif event.key == pygame.K_LEFT:
                self.cursor_pos = max(0, self.cursor_pos - 1)
            elif event.key == pygame.K_RIGHT:
                self.cursor_pos = min(len(self.text), self.cursor_pos + 1)
            elif event.key == pygame.K_HOME:
                self.cursor_pos = 0
            elif event.key == pygame.K_END:
                self.cursor_pos = len(self.text)
            elif event.unicode and event.unicode.isprintable():
                self.text = self.text[:self.cursor_pos] + event.unicode + self.text[self.cursor_pos:]
                self.cursor_pos += 1
            
            # Emit change signal if text changed
            if self.text != old_text:
                self.emit_signal('change', {'text': self.text, 'old_text': old_text})
                self.emit_signal('on_change', self.text)
                # Update value port
                port = self.get_port('value')
                if port:
                    port.value = self.text
            return True
        return False
        
    def update(self, dt: float):
        """Update cursor blink."""
        self.cursor_timer += dt
        if self.cursor_timer > 0.5:
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = 0.0
            
    def draw(self, surface: pygame.Surface):
        """Draw text input with console style."""
        if not self.visible:
            return
        
        # Draw background
        bg_color = DesignSystem.COLORS['surface_active'] if self.active else DesignSystem.COLORS['surface_light']
        pygame.draw.rect(surface, bg_color, self.rect,
                       border_radius=DesignSystem.RADIUS['sm'])
        
        # Draw border with glow when active
        border_color = DesignSystem.COLORS['primary'] if self.active else DesignSystem.COLORS['border']
        pygame.draw.rect(surface, border_color, self.rect,
                       width=2 if self.active else 1, border_radius=DesignSystem.RADIUS['sm'])
        
        # Draw text
        font = DesignSystem.get_font('console')
        display_text = self.text if self.text else self.placeholder
        text_color = DesignSystem.COLORS['text'] if self.text else DesignSystem.COLORS['text_tertiary']
        
        # Clip text if too long
        text_surf = font.render(display_text, True, text_color)
        clip_rect = self.rect.inflate(-DesignSystem.SPACING['md'], 0)
        old_clip = surface.get_clip()
        surface.set_clip(clip_rect)
        
        text_y = self.rect.y + (self.rect.height - text_surf.get_height()) // 2
        surface.blit(text_surf, (clip_rect.x, text_y))
        
        # Draw cursor
        if self.active and self.cursor_visible:
            cursor_text = self.text[:self.cursor_pos]
            cursor_surf = font.render(cursor_text, True, text_color)
            cursor_x = clip_rect.x + cursor_surf.get_width()
            pygame.draw.line(surface, DesignSystem.COLORS['primary'],
                           (cursor_x, self.rect.y + 4),
                           (cursor_x, self.rect.bottom - 4), 2)
        
        surface.set_clip(old_clip)


class Checkbox(UIComponent):
    """Checkbox component."""
    
    def __init__(self, x: int, y: int, text: str, checked: bool = False,
                 callback: Callable = None):
        super().__init__(x, y, 20, 20)
        self.text = text
        self.checked = checked
        self.hovered = False
        
        # Connect callback to port system if provided
        if callback:
            self.connect_port('on_change', callback)
        
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle checkbox events with port system integration."""
        if not self.visible or not self.enabled:
            return False
            
        if event.type == pygame.MOUSEMOTION:
            was_hovered = self.hovered
            self.hovered = self.rect.collidepoint(event.pos) if hasattr(event, 'pos') else False
            if was_hovered != self.hovered:
                self.emit_signal('hover', self.hovered)
                self.emit_signal('on_hover', {'hovered': self.hovered, 'text': self.text})
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if hasattr(event, 'pos') and self.rect.collidepoint(event.pos):
                old_checked = self.checked
                self.checked = not self.checked
                # Emit signals through port system
                self.emit_signal('click', {'checked': self.checked})
                self.emit_signal('change', {'checked': self.checked, 'old_checked': old_checked})
                self.emit_signal('on_change', self.checked)
                # Update value port
                port = self.get_port('value')
                if port:
                    port.value = self.checked
                return True
        return False
        
    def draw(self, surface: pygame.Surface):
        """Draw checkbox."""
        if not self.visible:
            return
            
        # Draw box
        bg_color = DesignSystem.COLORS['primary'] if self.checked else DesignSystem.COLORS['surface_light']
        pygame.draw.rect(surface, bg_color, self.rect,
                       border_radius=DesignSystem.RADIUS['sm'])
        pygame.draw.rect(surface, DesignSystem.COLORS['border_light'], self.rect,
                       width=1, border_radius=DesignSystem.RADIUS['sm'])
        
        # Draw checkmark
        if self.checked:
            points = [
                (self.rect.x + 5, self.rect.y + 10),
                (self.rect.x + 9, self.rect.y + 14),
                (self.rect.x + 15, self.rect.y + 6)
            ]
            pygame.draw.lines(surface, DesignSystem.COLORS['text'], False, points, 2)
        
        # Draw label
        font = DesignSystem.get_font('label')
        label_surf = font.render(self.text, True, DesignSystem.COLORS['text'])
        label_y = self.rect.y + (self.rect.height - label_surf.get_height()) // 2
        surface.blit(label_surf, (self.rect.right + DesignSystem.SPACING['sm'], label_y))


class Items(UIComponent):
    """List items component with selection."""
    
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        self.items: List[Tuple[str, str]] = []  # List of (name, type) tuples
        self.selected_index = -1
        self.scroll_y = 0
        self.item_height = 32
        self.on_select: Optional[Callable] = None
        
    def set_items(self, items: List[Tuple[str, str]]):
        """Set items list."""
        if items and isinstance(items[0], str):
            self.items = [(item, "") for item in items]
        else:
            self.items = items
            
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle items list events."""
        if not self.visible or not self.enabled:
            return False
            
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                rel_y = event.pos[1] - self.rect.y
                index = (rel_y + self.scroll_y) // self.item_height
                if 0 <= index < len(self.items):
                    self.selected_index = index
                    if self.on_select:
                        self.on_select(self.items[index])
                    return True
            elif event.button == 4:  # Scroll up
                self.scroll_y = max(0, self.scroll_y - 20)
            elif event.button == 5:  # Scroll down
                self.scroll_y += 20
        return False
        
    def draw(self, surface: pygame.Surface):
        """Draw items list."""
        if not self.visible:
            return
            
        # Draw background
        pygame.draw.rect(surface, DesignSystem.COLORS['surface'], self.rect,
                       border_radius=DesignSystem.RADIUS['md'])
        pygame.draw.rect(surface, DesignSystem.COLORS['border'], self.rect,
                       width=1, border_radius=DesignSystem.RADIUS['md'])
        
        # Clip to item area
        clip_rect = self.rect.inflate(-DesignSystem.SPACING['sm'], -DesignSystem.SPACING['sm'])
        old_clip = surface.get_clip()
        surface.set_clip(clip_rect)
        
        font = DesignSystem.get_font('console')
        small_font = DesignSystem.get_font('small')
        
        y_offset = clip_rect.y - self.scroll_y
        for i, (name, item_type) in enumerate(self.items):
            item_y = y_offset + i * self.item_height
            if item_y + self.item_height < clip_rect.y:
                continue
            if item_y > clip_rect.bottom:
                break
                
            item_rect = pygame.Rect(clip_rect.x, item_y, clip_rect.width, self.item_height)
            
            # Highlight selected
            if i == self.selected_index:
                pygame.draw.rect(surface, DesignSystem.COLORS['primary'], item_rect,
                               border_radius=DesignSystem.RADIUS['sm'])
                text_color = DesignSystem.COLORS['text']
            else:
                text_color = DesignSystem.COLORS['text_secondary']
            
            # Draw item name
            name_surf = font.render(name, True, text_color)
            surface.blit(name_surf, (item_rect.x + DesignSystem.SPACING['sm'], 
                                   item_rect.y + (self.item_height - name_surf.get_height()) // 2))
            
            # Draw item type if available
            if item_type:
                type_surf = small_font.render(item_type, True, DesignSystem.COLORS['text_tertiary'])
                type_y = item_rect.y + name_surf.get_height() + 2
                if type_y + type_surf.get_height() < item_rect.bottom:
                    surface.blit(type_surf, (item_rect.x + DesignSystem.SPACING['sm'], type_y))
        
        surface.set_clip(old_clip)


# ============================================================================
# PROFESSIONAL DISPLAY COMPONENTS
# ============================================================================

class ImageDisplayComponent(UIComponent):
    """Professional image display component with optimized rendering."""
    
    def __init__(self, x: int, y: int, width: int, height: int, title: str = ""):
        super().__init__(x, y, width, height)
        self.title = title
        self.image: Optional[pygame.Surface] = None
        self.placeholder_text = "Waiting for image..."
        
    def set_image(self, image: Optional[pygame.Surface]):
        """Set the image to display."""
        self.image = image
        
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle events."""
        if not self.visible or not self.enabled:
            return False
        return self.rect.collidepoint(event.pos) if hasattr(event, 'pos') else False
        
    def update(self, dt: float):
        """Update component state."""
        pass
        
    def draw(self, surface: pygame.Surface):
        """Draw image display component."""
        if not self.visible:
            return
            
        # Draw background
        pygame.draw.rect(surface, DesignSystem.COLORS['surface'], self.rect,
                        border_radius=DesignSystem.RADIUS['lg'])
        pygame.draw.rect(surface, DesignSystem.COLORS['border'], self.rect,
                        width=1, border_radius=DesignSystem.RADIUS['lg'])
        
        # Draw title if present
        if self.title:
            header_rect = pygame.Rect(self.rect.x, self.rect.y, self.rect.width, 36)
            pygame.draw.rect(surface, DesignSystem.COLORS['surface_light'], header_rect,
                           border_radius=DesignSystem.RADIUS['lg'])
            pygame.draw.rect(surface, DesignSystem.COLORS['border_light'], header_rect,
                           width=1, border_radius=DesignSystem.RADIUS['lg'])
            
            font = DesignSystem.get_font('label')
            title_surf = font.render(self.title, True, DesignSystem.COLORS['text'])
            title_y = header_rect.y + (header_rect.height - title_surf.get_height()) // 2
            surface.blit(title_surf, (header_rect.x + DesignSystem.SPACING['md'], title_y))
        
        # Calculate image area (below title if present)
        img_area = pygame.Rect(
            self.rect.x + 10,
            self.rect.y + (36 + 10 if self.title else 10),
            self.rect.width - 20,
            self.rect.height - (36 + 20 if self.title else 20)
        )
        
        # Draw image or placeholder
        if self.image:
            img_rect = self.image.get_rect()
            # Scale to fill area while maintaining aspect ratio
            scale = min(img_area.width / img_rect.width, img_area.height / img_rect.height)
            new_size = (int(img_rect.width * scale), int(img_rect.height * scale))
            scaled_img = pygame.transform.scale(self.image, new_size)
            scaled_rect = scaled_img.get_rect(center=img_area.center)
            surface.blit(scaled_img, scaled_rect)
        else:
            # Draw placeholder
            font = DesignSystem.get_font('label')
            placeholder_surf = font.render(self.placeholder_text, True, 
                                          DesignSystem.COLORS['text_secondary'])
            placeholder_rect = placeholder_surf.get_rect(center=img_area.center)
            surface.blit(placeholder_surf, placeholder_rect)


class PointCloudDisplayComponent(UIComponent):
    """Professional point cloud display component with 3D view controls."""
    
    def __init__(self, x: int, y: int, width: int, height: int, title: str = ""):
        super().__init__(x, y, width, height)
        self.title = title
        self.pc_surface: Optional[pygame.Surface] = None
        self.renderer: Optional[PointCloudRenderer] = None
        
        # Camera controls
        self.camera_angle_x = 0.0
        self.camera_angle_y = 0.0
        self.zoom = 1.0
        
        # 3D cube control (bottom left corner)
        self.cube_size = 80
        self.cube_margin = 15
        self.cube_hovered_face = None
        self.cube_rotation = 0.0
        
        # Drag state for cube rotation
        self.cube_dragging = False
        self.cube_drag_start_pos = None
        self.cube_drag_start_angles = None
        self.cube_drag_initial_pos = None
        self.cube_drag_initial_angles = None
        
        # View presets
        self.view_presets = {
            'front': (0.0, 0.0),
            'top': (-math.pi / 2, 0.0),
            'side': (0.0, math.pi / 2),
            'iso': (-math.pi / 6, math.pi / 4),
            'back': (0.0, math.pi),
            'bottom': (math.pi / 2, 0.0),
        }
        
        if HAS_POINTCLOUD:
            self.renderer = PointCloudRenderer(width=width - 20, height=height - 56)
        
    def set_pointcloud(self, pc_surface: Optional[pygame.Surface]):
        """Set the point cloud surface to display."""
        self.pc_surface = pc_surface
        
    def set_camera(self, angle_x: float, angle_y: float, zoom: float):
        """Set camera parameters with port system notification."""
        old_camera = (self.camera_angle_x, self.camera_angle_y, self.zoom)
        self.camera_angle_x = angle_x
        self.camera_angle_y = angle_y
        self.zoom = zoom
        if self.renderer:
            self.renderer.set_camera(angle_x, angle_y, zoom)
        
        # Emit camera change signal through port system
        new_camera = (angle_x, angle_y, zoom)
        self.emit_signal('change', {'camera': new_camera, 'old_camera': old_camera})
        self.emit_signal('on_change', new_camera)
        # Update value port
        port = self.get_port('value')
        if port:
            port.value = new_camera
            
    def get_camera(self) -> Tuple[float, float, float]:
        """Get current camera parameters."""
        return (self.camera_angle_x, self.camera_angle_y, self.zoom)
        
    def _get_cube_rect(self) -> pygame.Rect:
        """Get the 3D cube control rect (bottom left) - relative to component origin (0,0)."""
        # Return rect relative to component's origin, not absolute screen coordinates
        return pygame.Rect(
            self.cube_margin,
            self.rect.height - self.cube_size - self.cube_margin,
            self.cube_size,
            self.cube_size
        )
    
    def _get_cube_geometry(self):
        """Get cube geometry (vertices, faces, normals) in 3D space."""
        if not HAS_NUMPY:
            return None, None, None
            
        size = self.cube_size * 0.25
        
        # Cube vertices in 3D space (relative to center)
        vertices_3d = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Back face (0-3)
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],     # Front face (4-7)
        ], dtype=np.float32) * size
        
        # Define faces with vertex indices and their normals (before rotation)
        faces_data = [
            ([4, 5, 6, 7], 'front', np.array([0, 0, 1])),   # Front face (positive Z)
            ([0, 3, 2, 1], 'back', np.array([0, 0, -1])),   # Back face (negative Z)
            ([3, 7, 6, 2], 'top', np.array([0, 1, 0])),     # Top face (positive Y)
            ([0, 1, 5, 4], 'bottom', np.array([0, -1, 0])),  # Bottom face (negative Y)
            ([1, 2, 6, 5], 'right', np.array([1, 0, 0])),    # Right face (positive X)
            ([0, 4, 7, 3], 'left', np.array([-1, 0, 0])),   # Left face (negative X)
        ]
        
        # Apply rotation to vertices and normals
        cos_x, sin_x = math.cos(self.camera_angle_x), math.sin(self.camera_angle_x)
        cos_y, sin_y = math.cos(self.camera_angle_y), math.sin(self.camera_angle_y)
        
        # Rotate vertices
        rotated_vertices = []
        for v in vertices_3d:
            x, y, z = v[0], v[1], v[2]
            y_rot = y * cos_x - z * sin_x
            z_rot = y * sin_x + z * cos_x
            x_final = x * cos_y + z_rot * sin_y
            z_final = -x * sin_y + z_rot * cos_y
            rotated_vertices.append(np.array([x_final, y_rot, z_final]))
        
        # Rotate normals
        rotated_faces = []
        for face_indices, face_name, normal in faces_data:
            # Rotate normal vector
            nx, ny, nz = normal[0], normal[1], normal[2]
            ny_rot = ny * cos_x - nz * sin_x
            nz_rot = ny * sin_x + nz * cos_x
            nx_final = nx * cos_y + nz_rot * sin_y
            nz_final = -nx * sin_y + nz_rot * cos_y
            rotated_normal = np.array([nx_final, ny_rot, nz_final])
            rotated_faces.append((face_indices, face_name, rotated_normal))
        
        return rotated_vertices, rotated_faces, (cos_x, sin_x, cos_y, sin_y)
    
    def _project_3d_to_2d(self, vertex_3d, center_x, center_y):
        """Project a 3D vertex to 2D screen coordinates."""
        x, y, z = vertex_3d[0], vertex_3d[1], vertex_3d[2]
        # Simple orthographic projection
        proj_x = int(center_x + x)
        proj_y = int(center_y - y)  # Flip Y axis
        return (proj_x, proj_y)
    
    def _get_face_from_mouse(self, pos: Tuple[int, int]) -> Optional[str]:
        """Accurately determine which cube face is under the mouse using improved 3D ray casting."""
        if not HAS_NUMPY:
            return None
            
        cube_rect = self._get_cube_rect()
        if not cube_rect.collidepoint(pos):
            return None
        
        center_x, center_y = cube_rect.center
        rotated_vertices, rotated_faces, _ = self._get_cube_geometry()
        if rotated_vertices is None:
            return None
        
        # Convert mouse position to relative coordinates (relative to cube center)
        rel_x = pos[0] - center_x
        rel_y = pos[1] - center_y
        
        # Project all faces to 2D and find which one contains the mouse point
        # This is more reliable than 3D ray casting for orthographic projection
        best_face = None
        best_depth = float('inf')
        best_distance = float('inf')
        
        for face_indices, face_name, face_normal in rotated_faces:
            # Only consider faces facing the camera
            if face_normal[2] >= 0:  # Face is not visible (back-facing)
                continue
            
            # Project face vertices to 2D screen coordinates
            face_2d = [self._project_3d_to_2d(rotated_vertices[idx], center_x, center_y) for idx in face_indices]
            mouse_2d = (pos[0], pos[1])
            
            # Check if mouse point is inside the projected face polygon
            if self._point_in_polygon(mouse_2d, face_2d):
                # Calculate face center depth for z-ordering
                face_center = np.mean([rotated_vertices[idx] for idx in face_indices], axis=0)
                depth = face_center[2]  # Z depth (negative is closer to camera)
                
                # Also calculate distance from mouse to face center in 2D for tie-breaking
                face_center_2d = self._project_3d_to_2d(face_center, center_x, center_y)
                distance = ((mouse_2d[0] - face_center_2d[0])**2 + (mouse_2d[1] - face_center_2d[1])**2)**0.5
                
                # Prefer closer faces (more negative depth), and if same depth, prefer closer in 2D
                if depth < best_depth or (abs(depth - best_depth) < 0.1 and distance < best_distance):
                    best_depth = depth
                    best_distance = distance
                    best_face = face_name
        
        return best_face
    
    def _point_in_polygon(self, point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
        """Check if a point is inside a polygon using improved ray casting algorithm."""
        if len(polygon) < 3:
            return False
            
        x, y = point
        n = len(polygon)
        inside = False
        
        # Use ray casting algorithm with proper edge handling
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            # Check if point is on the edge (with small tolerance)
            if abs((yj - yi) * (x - xi) - (xj - xi) * (y - yi)) < 1e-6:
                # Point is on the line segment
                if min(xi, xj) <= x <= max(xi, xj) and min(yi, yj) <= y <= max(yi, yj):
                    return True
            
            # Ray casting: check if ray from point crosses edge
            if ((yi > y) != (yj > y)):  # Edge crosses horizontal line through point
                # Calculate x-coordinate of intersection
                if yj != yi:  # Avoid division by zero
                    x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
                    if x < x_intersect:
                        inside = not inside
            
            j = i
        
        return inside
        
    def _draw_3d_cube_control(self, surface: pygame.Surface):
        """Draw 3D cube control with accurate face detection and hover highlighting."""
        if not HAS_NUMPY:
            return
            
        cube_rect = self._get_cube_rect()
        
        # Draw cube background with hover effect
        bg_color = DesignSystem.COLORS['surface_active'] if self.cube_hovered_face else DesignSystem.COLORS['surface_light']
        pygame.draw.rect(surface, bg_color, cube_rect,
                       border_radius=DesignSystem.RADIUS['sm'])
        border_color = DesignSystem.COLORS['primary'] if self.cube_hovered_face else DesignSystem.COLORS['border']
        pygame.draw.rect(surface, border_color, cube_rect,
                       width=2 if self.cube_hovered_face else 1, border_radius=DesignSystem.RADIUS['sm'])
        
        # Get cube geometry
        rotated_vertices, rotated_faces, _ = self._get_cube_geometry()
        if rotated_vertices is None:
            return
        
        center_x, center_y = cube_rect.center
        
        # Project vertices to 2D
        vertices_2d = [self._project_3d_to_2d(v, center_x, center_y) for v in rotated_vertices]
        
        # Define face colors (RGB)
        base_face_colors = {
            'front': (255, 0, 0),      # Red
            'back': (0, 255, 0),       # Green
            'top': (0, 0, 255),        # Blue
            'bottom': (255, 255, 0),   # Yellow
            'right': (255, 0, 255),    # Magenta
            'left': (0, 255, 255),     # Cyan
        }
        
        # Calculate face depths and prepare for rendering
        face_render_data = []
        for face_indices, face_name, face_normal in rotated_faces:
            # Calculate average depth
            face_center = np.mean([rotated_vertices[idx] for idx in face_indices], axis=0)
            depth = face_center[2]
            
            # Only render faces facing camera (normal.z < 0)
            if face_normal[2] < 0:
                face_points = [vertices_2d[idx] for idx in face_indices]
                face_render_data.append((depth, face_indices, face_name, face_points))
        
        # Sort faces by depth (back to front)
        face_render_data.sort(key=lambda x: x[0], reverse=True)
        
        # Draw faces (back to front)
        for depth, face_indices, face_name, face_points in face_render_data:
            base_color = base_face_colors[face_name]
            
            # Highlight hovered face
            if self.cube_hovered_face == face_name:
                # Brighten and add glow effect
                highlight_color = tuple(min(255, int(c * 1.5)) for c in base_color)
                alpha_color = tuple(min(255, int(c * 0.9)) for c in highlight_color)
                border_width = 3
                border_color = highlight_color
            else:
                # Normal appearance
                alpha_color = tuple(int(c * 0.6) for c in base_color)
                border_width = 2
                border_color = base_color
            
            # Draw face fill
            pygame.draw.polygon(surface, alpha_color, face_points)
            # Draw face border
            pygame.draw.polygon(surface, border_color, face_points, width=border_width)
        
        # Draw cube edges for better definition
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Back face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Front face
            (0, 4), (1, 5), (2, 6), (3, 7),  # Connecting edges
        ]
        
        edge_color = DesignSystem.COLORS['primary'] if self.cube_hovered_face else DesignSystem.COLORS['text']
        for edge in edges:
            pygame.draw.line(surface, edge_color,
                           vertices_2d[edge[0]], vertices_2d[edge[1]], 1)
        
        # Draw face labels
        font = DesignSystem.get_font('small')
        # Determine which face is facing forward based on camera angles
        if abs(self.camera_angle_x + math.pi / 2) < 0.3:
            label = "TOP"
        elif abs(self.camera_angle_x - math.pi / 2) < 0.3:
            label = "BOT"
        elif abs(self.camera_angle_y) < 0.3:
            label = "FRONT"
        elif abs(self.camera_angle_y - math.pi) < 0.3 or abs(self.camera_angle_y + math.pi) < 0.3:
            label = "BACK"
        elif abs(self.camera_angle_y - math.pi / 2) < 0.3:
            label = "RIGHT"
        elif abs(self.camera_angle_y + math.pi / 2) < 0.3:
            label = "LEFT"
        else:
            label = "ISO"
            
        label_color = DesignSystem.COLORS['primary'] if self.cube_hovered_face else DesignSystem.COLORS['text']
        label_surf = font.render(label, True, label_color)
        label_rect = label_surf.get_rect(center=(center_x, center_y))
        surface.blit(label_surf, label_rect)
        
        # Draw instruction hint
        hint_font = DesignSystem.get_font('small')
        hint_text = "Click face to rotate"
        hint_surf = hint_font.render(hint_text, True, DesignSystem.COLORS['text_tertiary'])
        hint_y = cube_rect.bottom + 5
        if hint_y + hint_surf.get_height() < self.rect.bottom:
            surface.blit(hint_surf, (cube_rect.x, hint_y))
            
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle events including cube control clicks and drag rotation with port system integration."""
        if not self.visible or not self.enabled:
            return False
            
        cube_rect = self._get_cube_rect()
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if hasattr(event, 'pos') and cube_rect.collidepoint(event.pos):
                # Check if it's a drag (middle button or right button) or click (left button)
                if event.button == 1:  # Left button - click to set view
                    # Use accurate face detection
                    face = self._get_face_from_mouse(event.pos)
                    
                    if face:
                        # Map face to view preset
                        view_map = {
                            'front': 'front',
                            'back': 'back',
                            'top': 'top',
                            'bottom': 'bottom',
                            'right': 'side',
                            'left': 'side',  # Both left and right map to side view
                        }
                        preset = view_map.get(face, 'iso')
                        
                        if preset in self.view_presets:
                            angle_x, angle_y = self.view_presets[preset]
                            self.set_camera(angle_x, angle_y, 1.0)
                            
                            # Emit signals through port system
                            self.emit_signal('click', face)
                            self.emit_signal('on_click', {'face': face, 'preset': preset, 
                                                           'camera': (angle_x, angle_y, 1.0)})
                            self.emit_signal('change', {'camera': (angle_x, angle_y, 1.0)})
                            return True
                    else:
                        # Click on cube but not on a face - cycle views
                        current_preset = None
                        for name, (ax, ay) in self.view_presets.items():
                            if abs(self.camera_angle_x - ax) < 0.1 and abs(self.camera_angle_y - ay) < 0.1:
                                current_preset = name
                                break
                        
                        # Cycle to next preset
                        preset_order = ['front', 'top', 'side', 'iso', 'back', 'bottom']
                        if current_preset in preset_order:
                            current_idx = preset_order.index(current_preset)
                            next_idx = (current_idx + 1) % len(preset_order)
                            next_preset = preset_order[next_idx]
                        else:
                            next_preset = 'front'
                        
                        if next_preset in self.view_presets:
                            angle_x, angle_y = self.view_presets[next_preset]
                            self.set_camera(angle_x, angle_y, 1.0)
                            self.emit_signal('click', 'cycle')
                            self.emit_signal('on_click', {'action': 'cycle', 'preset': next_preset})
                            return True
                elif event.button in (2, 3):  # Middle or right button - start drag
                    # Start dragging for rotation
                    self.cube_dragging = True
                    self.cube_drag_start_pos = event.pos
                    self.cube_drag_initial_pos = event.pos
                    self.cube_drag_start_angles = (self.camera_angle_x, self.camera_angle_y)
                    self.cube_drag_initial_angles = (self.camera_angle_x, self.camera_angle_y)
                    return True
                    
        elif event.type == pygame.MOUSEBUTTONUP:
            if self.cube_dragging:
                self.cube_dragging = False
                self.cube_drag_start_pos = None
                self.cube_drag_initial_pos = None
                self.cube_drag_start_angles = None
                self.cube_drag_initial_angles = None
                return True
                    
        elif event.type == pygame.MOUSEMOTION:
            if self.cube_dragging and self.cube_drag_initial_pos and self.cube_drag_initial_angles:
                # Calculate rotation based on mouse movement
                if hasattr(event, 'pos'):
                    # Calculate movement from initial drag position
                    if hasattr(event, 'rel') and event.rel:
                        # Use relative movement for smooth rotation (preferred)
                        dx = event.rel[0]
                        dy = event.rel[1]
                        # Update start angles for next frame (accumulate)
                        new_angle_y = self.cube_drag_start_angles[1] + dx * 0.01
                        new_angle_x = self.cube_drag_start_angles[0] - dy * 0.01
                        self.cube_drag_start_angles = (new_angle_x, new_angle_y)
                    else:
                        # Fallback: calculate from initial position
                        dx = event.pos[0] - self.cube_drag_initial_pos[0]
                        dy = event.pos[1] - self.cube_drag_initial_pos[1]
                        sensitivity = 0.01
                        new_angle_y = self.cube_drag_initial_angles[1] + dx * sensitivity
                        new_angle_x = self.cube_drag_initial_angles[0] - dy * sensitivity
                    
                    # Clamp X angle to prevent flipping
                    new_angle_x = max(-math.pi / 2, min(math.pi / 2, new_angle_x))
                    
                    self.set_camera(new_angle_x, new_angle_y, self.zoom)
                    return True
            elif hasattr(event, 'pos') and cube_rect.collidepoint(event.pos):
                # Use accurate face detection for hover (only when not dragging)
                if not self.cube_dragging:
                    hovered_face = self._get_face_from_mouse(event.pos)
                    if hovered_face != self.cube_hovered_face:
                        self.cube_hovered_face = hovered_face
                        # Emit hover signal
                        self.emit_signal('hover', hovered_face)
                        self.emit_signal('on_hover', {'face': hovered_face})
            else:
                if self.cube_hovered_face is not None:
                    self.cube_hovered_face = None
                    self.emit_signal('hover', None)
            
        return False
        
    def update(self, dt: float):
        """Update component state."""
        self.cube_rotation += dt * 0.5  # Slow rotation animation
        
    def draw(self, surface: pygame.Surface):
        """Draw point cloud display component with enhanced visuals."""
        if not self.visible:
            return
            
        # Draw background with subtle gradient effect
        pygame.draw.rect(surface, DesignSystem.COLORS['bg_panel'], self.rect,
                        border_radius=DesignSystem.RADIUS['md'])
        pygame.draw.rect(surface, DesignSystem.COLORS['border'], self.rect,
                        width=1, border_radius=DesignSystem.RADIUS['md'])
        
        # Draw title if present
        # Note: Card sets child.rect to (0,0) during draw, so use relative coordinates
        if self.title:
            header_rect = pygame.Rect(0, 0, self.rect.width, 36)
            pygame.draw.rect(surface, DesignSystem.COLORS['surface_light'], header_rect,
                           border_radius=DesignSystem.RADIUS['md'])
            pygame.draw.rect(surface, DesignSystem.COLORS['border_light'], header_rect,
                           width=1, border_radius=DesignSystem.RADIUS['md'])
            
            font = DesignSystem.get_font('label')
            title_surf = font.render(self.title, True, DesignSystem.COLORS['text'])
            title_y = header_rect.y + (header_rect.height - title_surf.get_height()) // 2
            surface.blit(title_surf, (header_rect.x + DesignSystem.SPACING['md'], title_y))
        
        # Calculate point cloud area (below title if present, with padding)
        # Use relative coordinates since Card sets rect to (0,0) during draw
        padding = DesignSystem.SPACING['sm']
        header_height = 36 if self.title else 0
        pc_area = pygame.Rect(
            padding,
            header_height + padding,
            self.rect.width - padding * 2,
            self.rect.height - header_height - padding * 2
        )
        
        # Draw point cloud or placeholder
        if self.pc_surface:
            try:
                pc_copy = self.pc_surface.copy()
                pc_rect = pc_copy.get_rect()
                if pc_rect.width > 0 and pc_rect.height > 0:
                    # Scale to fill area completely while maintaining aspect ratio
                    scale_w = pc_area.width / pc_rect.width
                    scale_h = pc_area.height / pc_rect.height
                    scale = max(scale_w, scale_h)  # Fill entire area
                    new_size = (int(pc_rect.width * scale), int(pc_rect.height * scale))
                    if new_size[0] > 0 and new_size[1] > 0:
                        scaled_pc = pygame.transform.scale(pc_copy, new_size)
                        scaled_rect = scaled_pc.get_rect(center=pc_area.center)
                        surface.blit(scaled_pc, scaled_rect)
            except Exception:
                pass
        else:
            # Draw enhanced placeholder with icon-like visual
            center_x, center_y = pc_area.center
            
            # Draw placeholder background
            placeholder_bg = pygame.Rect(
                center_x - 150, center_y - 40,
                300, 80
            )
            pygame.draw.rect(surface, DesignSystem.COLORS['surface_light'], placeholder_bg,
                           border_radius=DesignSystem.RADIUS['md'])
            pygame.draw.rect(surface, DesignSystem.COLORS['border'], placeholder_bg,
                           width=1, border_radius=DesignSystem.RADIUS['md'])
            
            # Draw placeholder text
            font = DesignSystem.get_font('label')
            placeholder_surf = font.render("Waiting for point cloud data...", True,
                                          DesignSystem.COLORS['text_secondary'])
            placeholder_rect = placeholder_surf.get_rect(center=(center_x, center_y))
            surface.blit(placeholder_surf, placeholder_rect)
        
        # Draw 3D cube control (bottom left) - only if there's space
        cube_rect = self._get_cube_rect()
        if cube_rect.bottom <= self.rect.bottom - padding:
            self._draw_3d_cube_control(surface)


# ============================================================================
# ADVANCED COMPONENTS
# ============================================================================

class MapComponent(UIComponent):
    """Professional map display component for showing drone positions and trajectories."""
    
    def __init__(self, x: int, y: int, width: int, height: int, title: str = ""):
        super().__init__(x, y, width, height)
        self.title = title
        self.drones: Dict[str, Dict[str, Any]] = {}  # Will be set from parent
        self.current_drone_id: Optional[str] = None  # Will be set from parent
        
    def set_drones(self, drones: Dict[str, Dict[str, Any]], current_drone_id: Optional[str]):
        """Set the drones data and current selection."""
        self.drones = drones
        self.current_drone_id = current_drone_id
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle events - can be used for clicking on drones to select them."""
        if not self.visible or not self.enabled:
            return False
        if hasattr(event, 'pos') and self.rect.collidepoint(event.pos):
            # Could implement click-to-select drone here
            return True
        return False
    
    def update(self, dt: float):
        """Update component state."""
        pass
    
    def draw(self, surface: pygame.Surface):
        """Draw map component with drone positions and trajectories."""
        if not self.visible:
            return
        
        # Calculate map area (relative to component origin, which Card sets to 0,0)
        padding = DesignSystem.SPACING['md']
        header_height = 36 if self.title else 0
        map_rect = pygame.Rect(
            padding,
            header_height + padding,
            self.rect.width - padding * 2,
            self.rect.height - header_height - padding * 2
        )
        
        # Draw map background
        pygame.draw.rect(surface, DesignSystem.COLORS['bg_secondary'], map_rect,
                       border_radius=DesignSystem.RADIUS['sm'])
        
        # Collect all drone positions
        drone_positions = []
        for drone_id, drone_info in self.drones.items():
            if drone_info.get('client') and drone_info.get('is_connected'):
                try:
                    pos = drone_info['client'].get_position()
                    if len(pos) >= 3 and pos[0] != 0.0 and pos[1] != 0.0:
                        drone_positions.append({
                            'id': drone_id,
                            'name': drone_info['name'],
                            'lat': pos[0],
                            'lon': pos[1],
                            'alt': pos[2],
                            'state': drone_info['client'].get_status(),
                            'trajectory': drone_info.get('trajectory', [])
                        })
                except:
                    pass
        
        if len(drone_positions) == 0:
            # No drones with valid positions
            font = DesignSystem.get_font('label')
            no_data_text = font.render("No drone positions available. Connect drones to see their positions on the map.",
                                     True, DesignSystem.COLORS['text_secondary'])
            text_rect = no_data_text.get_rect(center=map_rect.center)
            surface.blit(no_data_text, text_rect)
        else:
            # Calculate map bounds
            lats = [d['lat'] for d in drone_positions]
            lons = [d['lon'] for d in drone_positions]
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)
            
            # Add padding
            lat_range = max(max_lat - min_lat, 0.001)  # Minimum range
            lon_range = max(max_lon - min_lon, 0.001)
            padding_factor = 0.1
            min_lat -= lat_range * padding_factor
            max_lat += lat_range * padding_factor
            min_lon -= lon_range * padding_factor
            max_lon += lon_range * padding_factor
            lat_range = max_lat - min_lat
            lon_range = max_lon - min_lon
            
            # Draw trajectories
            for drone_data in drone_positions:
                if len(drone_data['trajectory']) > 1:
                    points = []
                    for lat, lon, alt in drone_data['trajectory']:
                        x = map_rect.x + int((lon - min_lon) / lon_range * map_rect.width)
                        y = map_rect.y + int((max_lat - lat) / lat_range * map_rect.height)
                        points.append((x, y))
                    
                    if len(points) > 1:
                        # Draw trajectory line
                        color = DesignSystem.COLORS['primary']
                        if drone_data['id'] == self.current_drone_id:
                            color = DesignSystem.COLORS['accent']
                        pygame.draw.lines(surface, color, False, points, 2)
            
            # Draw drone positions
            font = DesignSystem.get_font('small')
            for drone_data in drone_positions:
                x = map_rect.x + int((drone_data['lon'] - min_lon) / lon_range * map_rect.width)
                y = map_rect.y + int((max_lat - drone_data['lat']) / lat_range * map_rect.height)
                
                # Draw drone icon (circle)
                color = DesignSystem.COLORS['success']
                if drone_data['id'] == self.current_drone_id:
                    color = DesignSystem.COLORS['accent']
                    # Draw larger circle for selected drone
                    pygame.draw.circle(surface, color, (x, y), 12, 2)
                
                # Draw drone position
                pygame.draw.circle(surface, color, (x, y), 8)
                pygame.draw.circle(surface, DesignSystem.COLORS['bg'], (x, y), 4)
                
                # Draw drone name and info
                state = drone_data['state']
                info_text = f"{drone_data['name']}"
                if state:
                    info_text += f" | Alt: {drone_data['alt']:.1f}m"
                    if state.armed:
                        info_text += " | ARMED"
                    if state.battery > 0:
                        info_text += f" | Bat: {state.battery:.0f}%"
                
                text_surf = font.render(info_text, True, DesignSystem.COLORS['text'])
                text_rect = text_surf.get_rect()
                text_rect.centerx = x
                text_rect.y = y + 15
                
                # Draw background for text
                bg_rect = text_rect.inflate(10, 5)
                pygame.draw.rect(surface, DesignSystem.COLORS['bg'], bg_rect,
                               border_radius=DesignSystem.RADIUS['sm'])
                pygame.draw.rect(surface, color, bg_rect, 1,
                               border_radius=DesignSystem.RADIUS['sm'])
                surface.blit(text_surf, text_rect)
            
            # Draw legend
            legend_y = map_rect.y + 10
            legend_x = map_rect.x + 10
            legend_font = DesignSystem.get_font('small')
            
            legend_items = [
                ("Selected Drone", DesignSystem.COLORS['accent']),
                ("Other Drones", DesignSystem.COLORS['success']),
                ("Trajectory", DesignSystem.COLORS['primary'])
            ]
            
            for i, (label, color) in enumerate(legend_items):
                pygame.draw.circle(surface, color, (legend_x + 5, legend_y + i * 20 + 5), 5)
                legend_text = legend_font.render(label, True, DesignSystem.COLORS['text'])
                surface.blit(legend_text, (legend_x + 15, legend_y + i * 20))


class JSONEditor(UIComponent):
    """Advanced JSON editor with syntax highlighting, selection, copy/paste, and more."""
    
    def __init__(self, x: int, y: int, width: int, height: int, default_text: str = ""):
        super().__init__(x, y, width, height)
        self.text = default_text
        self.active = False
        self.cursor_visible = True
        self.cursor_timer = 0.0
        self.scroll_y = 0
        self.scroll_x = 0
        self.cursor_pos = [0, 0]  # [line, col]
        self.selection_start = None  # [line, col] or None
        self.line_height = 18
        self.char_width = 8
        self.line_number_width = 50
        self.undo_stack = []  # History for undo
        self.undo_limit = 50
        self.clipboard = ""
        
    def _save_state(self):
        """Save current state for undo."""
        if len(self.undo_stack) == 0 or self.undo_stack[-1] != (self.text, self.cursor_pos):
            self.undo_stack.append((self.text, self.cursor_pos.copy()))
            if len(self.undo_stack) > self.undo_limit:
                self.undo_stack.pop(0)
    
    def _get_selected_text(self):
        """Get selected text."""
        if self.selection_start is None:
            return ""
        lines = self.text.split('\n')
        start_line, start_col = self.selection_start
        end_line, end_col = self.cursor_pos
        
        if start_line > end_line or (start_line == end_line and start_col > end_col):
            start_line, end_line = end_line, start_line
            start_col, end_col = end_col, start_col
        
        if start_line == end_line:
            return lines[start_line][start_col:end_col]
        else:
            result = [lines[start_line][start_col:]]
            for i in range(start_line + 1, end_line):
                result.append(lines[i])
            result.append(lines[end_line][:end_col])
            return '\n'.join(result)
    
    def _delete_selection(self):
        """Delete selected text."""
        if self.selection_start is None:
            return False
        
        lines = self.text.split('\n')
        start_line, start_col = self.selection_start
        end_line, end_col = self.cursor_pos
        
        if start_line > end_line or (start_line == end_line and start_col > end_col):
            start_line, end_line = end_line, start_line
            start_col, end_col = end_col, start_col
        
        if start_line == end_line:
            lines[start_line] = lines[start_line][:start_col] + lines[start_line][end_col:]
        else:
            lines[start_line] = lines[start_line][:start_col] + lines[end_line][end_col:]
            del lines[start_line + 1:end_line + 1]
        
        self.text = '\n'.join(lines)
        self.cursor_pos = [start_line, start_col]
        self.selection_start = None
        return True
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle JSON editor events with advanced editing capabilities."""
        if not self.visible or not self.enabled:
            return False
            
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = True
                # Calculate cursor position from mouse click
                rel_x = event.pos[0] - (self.rect.x + self.line_number_width + 8) + self.scroll_x
                rel_y = event.pos[1] - (self.rect.y + 4) + self.scroll_y
                click_line = max(0, int(rel_y / self.line_height))
                lines = self.text.split('\n')
                click_line = min(click_line, len(lines) - 1)
                click_col = max(0, min(int(rel_x / self.char_width), len(lines[click_line])))
                self.cursor_pos = [click_line, click_col]
                if event.button == 1:  # Left click
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        if self.selection_start is None:
                            self.selection_start = self.cursor_pos.copy()
                    else:
                        self.selection_start = None
            else:
                self.active = False
            if event.button == 4:  # Scroll up
                self.scroll_y = max(0, self.scroll_y - 20)
            elif event.button == 5:  # Scroll down
                self.scroll_y += 20
            return self.active
        elif event.type == pygame.KEYDOWN and self.active:
            # Handle modifier keys
            mods = pygame.key.get_mods()
            ctrl = mods & (pygame.KMOD_CTRL | pygame.KMOD_META)
            shift = mods & pygame.KMOD_SHIFT
            
            # Save state before modification
            if event.key not in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, 
                                pygame.K_HOME, pygame.K_END, pygame.K_PAGEUP, pygame.K_PAGEDOWN):
                self._save_state()
            
            lines = self.text.split('\n')
            
            # Copy/Cut/Paste
            if ctrl and event.key == pygame.K_c:
                selected = self._get_selected_text()
                if selected:
                    self.clipboard = selected
                return True
            elif ctrl and event.key == pygame.K_x:
                selected = self._get_selected_text()
                if selected:
                    self.clipboard = selected
                    self._delete_selection()
                return True
            elif ctrl and event.key == pygame.K_v:
                if self.clipboard:
                    if self._delete_selection():
                        lines = self.text.split('\n')
                    clipboard_lines = self.clipboard.split('\n')
                    if len(clipboard_lines) == 1:
                        lines[self.cursor_pos[0]] = (lines[self.cursor_pos[0]][:self.cursor_pos[1]] + 
                                                     clipboard_lines[0] + 
                                                     lines[self.cursor_pos[0]][self.cursor_pos[1]:])
                        self.cursor_pos[1] += len(clipboard_lines[0])
                    else:
                        line = lines[self.cursor_pos[0]]
                        lines[self.cursor_pos[0]] = line[:self.cursor_pos[1]] + clipboard_lines[0]
                        for i, clip_line in enumerate(clipboard_lines[1:], 1):
                            lines.insert(self.cursor_pos[0] + i, clip_line)
                        lines[self.cursor_pos[0] + len(clipboard_lines) - 1] += line[self.cursor_pos[1]:]
                        self.cursor_pos[0] += len(clipboard_lines) - 1
                        self.cursor_pos[1] = len(clipboard_lines[-1])
                    self.text = '\n'.join(lines)
                    self.selection_start = None
                return True
            elif ctrl and event.key == pygame.K_a:  # Select all
                self.selection_start = [0, 0]
                lines = self.text.split('\n')
                self.cursor_pos = [len(lines) - 1, len(lines[-1])]
                return True
            elif ctrl and event.key == pygame.K_z:  # Undo
                if self.undo_stack:
                    self.text, self.cursor_pos = self.undo_stack.pop()
                    self.selection_start = None
                return True
            
            # Delete key
            if event.key == pygame.K_DELETE:
                if self._delete_selection():
                    return True
                lines = self.text.split('\n')
                if self.cursor_pos[0] < len(lines):
                    if self.cursor_pos[1] < len(lines[self.cursor_pos[0]]):
                        lines[self.cursor_pos[0]] = (lines[self.cursor_pos[0]][:self.cursor_pos[1]] + 
                                                    lines[self.cursor_pos[0]][self.cursor_pos[1] + 1:])
                    elif self.cursor_pos[0] < len(lines) - 1:
                        lines[self.cursor_pos[0]] += lines.pop(self.cursor_pos[0] + 1)
                self.text = '\n'.join(lines)
                self.selection_start = None
                return True
            
            # Navigation with selection
            if event.key == pygame.K_UP:
                if shift and self.selection_start is None:
                    self.selection_start = self.cursor_pos.copy()
                self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
                self.cursor_pos[1] = min(self.cursor_pos[1], len(lines[self.cursor_pos[0]]))
                if not shift:
                    self.selection_start = None
                return True
            elif event.key == pygame.K_DOWN:
                if shift and self.selection_start is None:
                    self.selection_start = self.cursor_pos.copy()
                self.cursor_pos[0] = min(len(lines) - 1, self.cursor_pos[0] + 1)
                self.cursor_pos[1] = min(self.cursor_pos[1], len(lines[self.cursor_pos[0]]))
                if not shift:
                    self.selection_start = None
                return True
            elif event.key == pygame.K_LEFT:
                if shift and self.selection_start is None:
                    self.selection_start = self.cursor_pos.copy()
                if self.cursor_pos[1] > 0:
                    self.cursor_pos[1] -= 1
                elif self.cursor_pos[0] > 0:
                    self.cursor_pos[0] -= 1
                    self.cursor_pos[1] = len(lines[self.cursor_pos[0]])
                if not shift:
                    self.selection_start = None
                return True
            elif event.key == pygame.K_RIGHT:
                if shift and self.selection_start is None:
                    self.selection_start = self.cursor_pos.copy()
                if self.cursor_pos[0] < len(lines):
                    if self.cursor_pos[1] < len(lines[self.cursor_pos[0]]):
                        self.cursor_pos[1] += 1
                    elif self.cursor_pos[0] < len(lines) - 1:
                        self.cursor_pos[0] += 1
                        self.cursor_pos[1] = 0
                if not shift:
                    self.selection_start = None
                return True
            elif event.key == pygame.K_HOME:
                if shift and self.selection_start is None:
                    self.selection_start = self.cursor_pos.copy()
                self.cursor_pos[1] = 0
                if ctrl:
                    self.cursor_pos[0] = 0
                if not shift:
                    self.selection_start = None
                return True
            elif event.key == pygame.K_END:
                if shift and self.selection_start is None:
                    self.selection_start = self.cursor_pos.copy()
                self.cursor_pos[1] = len(lines[self.cursor_pos[0]])
                if ctrl:
                    self.cursor_pos[0] = len(lines) - 1
                    self.cursor_pos[1] = len(lines[-1])
                if not shift:
                    self.selection_start = None
                return True
            elif event.key == pygame.K_PAGEUP:
                self.scroll_y = max(0, self.scroll_y - self.rect.height // 2)
                return True
            elif event.key == pygame.K_PAGEDOWN:
                self.scroll_y += self.rect.height // 2
                return True
            
            # Text editing
            lines = self.text.split('\n')
            if event.key == pygame.K_BACKSPACE:
                if self._delete_selection():
                    return True
                if self.cursor_pos[0] < len(lines):
                    if self.cursor_pos[1] > 0:
                        lines[self.cursor_pos[0]] = (lines[self.cursor_pos[0]][:self.cursor_pos[1]-1] + 
                                                    lines[self.cursor_pos[0]][self.cursor_pos[1]:])
                        self.cursor_pos[1] -= 1
                    elif self.cursor_pos[0] > 0:
                        self.cursor_pos[1] = len(lines[self.cursor_pos[0]-1])
                        lines[self.cursor_pos[0]-1] += lines.pop(self.cursor_pos[0])
                        self.cursor_pos[0] -= 1
                    self.text = '\n'.join(lines)
                self.selection_start = None
                return True
            elif event.key == pygame.K_RETURN:
                if self._delete_selection():
                    lines = self.text.split('\n')
                if self.cursor_pos[0] < len(lines):
                    line = lines[self.cursor_pos[0]]
                    # Preserve indentation
                    indent = len(line) - len(line.lstrip())
                    lines[self.cursor_pos[0]] = line[:self.cursor_pos[1]]
                    lines.insert(self.cursor_pos[0] + 1, " " * indent + line[self.cursor_pos[1]:])
                    self.cursor_pos[0] += 1
                    self.cursor_pos[1] = indent
                    self.text = '\n'.join(lines)
                self.selection_start = None
                return True
            elif event.key == pygame.K_TAB:
                if self._delete_selection():
                    lines = self.text.split('\n')
                if self.cursor_pos[0] < len(lines):
                    lines[self.cursor_pos[0]] = (lines[self.cursor_pos[0]][:self.cursor_pos[1]] + 
                                                "    " + lines[self.cursor_pos[0]][self.cursor_pos[1]:])
                    self.cursor_pos[1] += 4
                    self.text = '\n'.join(lines)
                self.selection_start = None
                return True
            else:
                # Insert text
                if event.unicode and event.unicode.isprintable():
                    if self._delete_selection():
                        lines = self.text.split('\n')
                    if self.cursor_pos[0] < len(lines):
                        line = lines[self.cursor_pos[0]]
                        lines[self.cursor_pos[0]] = (line[:self.cursor_pos[1]] + event.unicode + 
                                                    line[self.cursor_pos[1]:])
                        self.cursor_pos[1] += 1
                        self.text = '\n'.join(lines)
                    self.selection_start = None
                    return True
            return True
        return False
                    
    def update(self, dt: float):
        """Update cursor blink."""
        self.cursor_timer += dt
        if self.cursor_timer > 0.5:
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = 0.0
            
    def format_json(self):
        """Format JSON text."""
        try:
            obj = json.loads(self.text)
            self.text = json.dumps(obj, indent=4)
            lines = self.text.split('\n')
            self.cursor_pos = [0, 0]
        except:
            pass
            
    def draw(self, surface: pygame.Surface):
        """Draw JSON editor with console style."""
        if not self.visible:
            return
        
        # Draw background
        bg_color = DesignSystem.COLORS['surface_active'] if self.active else DesignSystem.COLORS['surface']
        pygame.draw.rect(surface, bg_color, self.rect,
                       border_radius=DesignSystem.RADIUS['md'])
        border_color = DesignSystem.COLORS['primary'] if self.active else DesignSystem.COLORS['border']
        pygame.draw.rect(surface, border_color, self.rect,
                       width=2 if self.active else 1, border_radius=DesignSystem.RADIUS['md'])
        
        # Draw line numbers
        line_num_rect = pygame.Rect(self.rect.x + 4, self.rect.y + 4, 
                                   self.line_number_width, self.rect.height - 8)
        pygame.draw.rect(surface, DesignSystem.COLORS['bg'], line_num_rect,
                       border_radius=DesignSystem.RADIUS['sm'])
        
        # Clip to text area
        text_rect = pygame.Rect(
            self.rect.x + self.line_number_width + 8,
            self.rect.y + 4,
            self.rect.width - self.line_number_width - 12,
            self.rect.height - 8
        )
        old_clip = surface.get_clip()
        surface.set_clip(text_rect)
        
        font = DesignSystem.get_font('console')
        # Ensure text is always a string
        text_str = str(self.text) if self.text is not None else ""
        lines = text_str.split('\n') if text_str else [""]
        y_offset = text_rect.y - self.scroll_y
        
        # Draw selection highlight
        if self.selection_start is not None:
            start_line, start_col = self.selection_start
            end_line, end_col = self.cursor_pos
            
            if start_line > end_line or (start_line == end_line and start_col > end_col):
                start_line, end_line = end_line, start_line
                start_col, end_col = end_col, start_col
            
            for line_idx in range(start_line, end_line + 1):
                if 0 <= line_idx < len(lines):
                    line_y = y_offset + line_idx * self.line_height
                    if line_y + self.line_height < text_rect.y or line_y > text_rect.bottom:
                        continue
                    
                    if line_idx == start_line == end_line:
                        sel_x1 = text_rect.x + start_col * self.char_width
                        sel_x2 = text_rect.x + end_col * self.char_width
                    elif line_idx == start_line:
                        sel_x1 = text_rect.x + start_col * self.char_width
                        sel_x2 = text_rect.right
                    elif line_idx == end_line:
                        sel_x1 = text_rect.x
                        sel_x2 = text_rect.x + end_col * self.char_width
                    else:
                        sel_x1 = text_rect.x
                        sel_x2 = text_rect.right
                    
                    sel_rect = pygame.Rect(sel_x1, line_y, sel_x2 - sel_x1, self.line_height)
                    pygame.draw.rect(surface, DesignSystem.COLORS['primary'], sel_rect, 
                                   border_radius=2)
        
        for i, line in enumerate(lines):
            line_y = y_offset + i * self.line_height
            if line_y + self.line_height < text_rect.y:
                continue
            if line_y > text_rect.bottom:
                break
                
            # Draw line number
            line_num_surf = font.render(str(i + 1), True, DesignSystem.COLORS['text_tertiary'])
            surface.blit(line_num_surf, (self.rect.x + 8, line_y))
            
            # Draw line with syntax highlighting
            x_pos = text_rect.x - self.scroll_x
            for j, char in enumerate(line):
                char_x = x_pos + j * self.char_width
                if char_x + self.char_width < text_rect.x:
                    continue
                if char_x > text_rect.right:
                    break
                    
                # Enhanced syntax highlighting
                if char in ['{', '}', '[', ']']:
                    color = DesignSystem.COLORS['primary']
                elif char in [':', ',']:
                    color = DesignSystem.COLORS['text_secondary']
                elif char == '"':
                    color = DesignSystem.COLORS['success']
                elif char.isdigit() or char == '.' or char == '-':
                    color = DesignSystem.COLORS['warning']
                else:
                    color = DesignSystem.COLORS['text']
                    
                char_surf = font.render(char, True, color)
                surface.blit(char_surf, (char_x, line_y))
        
        # Draw cursor
        if self.active and self.cursor_visible:
            cursor_y = text_rect.y - self.scroll_y + self.cursor_pos[0] * self.line_height
            if text_rect.y <= cursor_y <= text_rect.bottom:
                lines = self.text.split('\n')
                cursor_x = text_rect.x - self.scroll_x + self.cursor_pos[1] * self.char_width
                if text_rect.x <= cursor_x <= text_rect.right:
                    pygame.draw.line(surface, DesignSystem.COLORS['text'],
                                       (cursor_x, cursor_y),
                                       (cursor_x, cursor_y + self.line_height), 2)
        
        surface.set_clip(old_clip)


class TopicList(Items):
    """Topic list component (specialized Items)."""
    
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)


# ============================================================================
# LAYOUT MANAGER
# ============================================================================

class LayoutManager:
    """Unified layout manager for automatic component sizing and padding calculation."""
    
    # Constants
    TAB_BAR_HEIGHT = 45
    CARD_TITLE_HEIGHT = 36
    COMPONENT_TITLE_HEIGHT = 36
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
    def get_content_area(self) -> pygame.Rect:
        """Get the main content area below tab bar with standard padding."""
        content_padding = DesignSystem.SPACING['md']
        start_y = self.TAB_BAR_HEIGHT + DesignSystem.SPACING['lg']
        return pygame.Rect(
            content_padding,
            start_y,
            self.screen_width - content_padding * 2,
            self.screen_height - start_y - content_padding
        )
    
    def calculate_header_area(self, content_area: pygame.Rect, 
                             title: str, subtitle: str = None) -> Tuple[pygame.Rect, int]:
        """Calculate header area (title + optional subtitle) and return header rect and total height."""
        y = content_area.y
        title_font = DesignSystem.get_font('title')
        title_height = title_font.get_height()
        
        # Calculate total header height
        header_height = title_height
        if subtitle:
            subtitle_font = DesignSystem.get_font('small')
            subtitle_height = subtitle_font.get_height()
            header_height += DesignSystem.SPACING['sm'] + subtitle_height
        
        header_height += DesignSystem.SPACING['md']  # Spacing after header
        
        header_rect = pygame.Rect(
            content_area.x,
            y,
            content_area.width,
            header_height
        )
        
        return header_rect, header_height
    
    def calculate_component_area(self, content_area: pygame.Rect, 
                                 header_height: int,
                                 min_height: int = 200) -> pygame.Rect:
        """Calculate component area below header with proper padding."""
        component_y = content_area.y + header_height
        component_height = content_area.height - header_height
        
        # Ensure minimum height
        component_height = max(min_height, component_height)
        
        # Ensure doesn't exceed screen
        max_height = self.screen_height - component_y - DesignSystem.SPACING['md']
        component_height = min(component_height, max_height)
        
        return pygame.Rect(
            content_area.x,
            component_y,
            content_area.width,
            component_height
        )
    
    def calculate_inner_content_area(self, component_rect: pygame.Rect,
                                     has_title: bool = True,
                                     padding: str = 'sm') -> pygame.Rect:
        """Calculate inner content area within a component (accounting for title and padding)."""
        padding_value = DesignSystem.SPACING[padding]
        title_offset = self.COMPONENT_TITLE_HEIGHT if has_title else 0
        
        return pygame.Rect(
            component_rect.x + padding_value,
            component_rect.y + title_offset + padding_value,
            component_rect.width - padding_value * 2,
            component_rect.height - title_offset - padding_value * 2
        )
    
    def calculate_indicator_position(self, component_rect: pygame.Rect,
                                    indicator_width: int = 90,
                                    indicator_height: int = 24) -> Optional[pygame.Rect]:
        """Calculate status indicator position (top-right of component, below title)."""
        indicator_y = component_rect.y + self.COMPONENT_TITLE_HEIGHT + DesignSystem.SPACING['sm']
        
        # Check if indicator fits within component
        if indicator_y + indicator_height <= component_rect.bottom:
            return pygame.Rect(
                self.screen_width - DesignSystem.SPACING['md'] - indicator_width,
                indicator_y,
                indicator_width,
                indicator_height
            )
        return None
    
    def calculate_renderer_size(self, inner_content_area: pygame.Rect,
                               min_size: int = 100) -> Tuple[int, int]:
        """Calculate renderer size from inner content area."""
        return (
            max(min_size, inner_content_area.width),
            max(min_size, inner_content_area.height)
        )


# ============================================================================
# MAIN GUI APPLICATION
# ============================================================================

class RosClientPygameGUI:
    """Industrial-grade Pygame GUI for RosClient with fighter cockpit design."""
    
    def __init__(self):
        pygame.init()
        DesignSystem.init_fonts()
        
        self.screen_width = 1400
        self.screen_height = 900
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("ROS Client - Fighter Cockpit Interface")
        
        self.clock = pygame.time.Clock()
        self.running = True
        self.dt = 0
        
        # Multi-drone client state
        self.drones: Dict[str, Dict[str, Any]] = {}  # drone_id -> {client, name, url, is_connected, image, pointcloud, trajectory}
        self.current_drone_id: Optional[str] = None  # Currently selected drone
        self.drone_counter = 0  # For generating unique IDs
        
        # Legacy single client support (for backward compatibility)
        self.client: Optional[RosClient] = None
        self.is_connected = False
        
        self.update_thread: Optional[threading.Thread] = None
        self.stop_update = threading.Event()
        self.image_queue = queue.Queue(maxsize=1)
        self.current_image = None
        
        # Tabs
        self.tabs = ["Connection", "Status", "Image", "Control", "Point Cloud", "3D View", "Map", "Network Test"]
        self.current_tab = 0
        
        # Layout manager for automatic sizing and padding
        self.layout = LayoutManager(self.screen_width, self.screen_height)
        
        # UI Components
        self.setup_ui()
        self.setup_update_loop()
        
    def setup_ui(self):
        """Setup UI components using new component system."""
        # Connection tab components - Multi-drone management
        self.drone_name_input = TextInput(200, 100, 200, 35, "Drone 1")
        self.connection_url_input = TextInput(420, 100, 400, 35, "ws://localhost:9090")
        self.use_mock_checkbox = Checkbox(200, 150, "Use Mock Client (Test Mode)", False)
        self.add_drone_btn = Button(200, 200, 120, 40, "Add Drone", self.add_drone)
        self.connect_btn = Button(330, 200, 120, 40, "Connect", self.connect)
        self.disconnect_btn = Button(460, 200, 120, 40, "Disconnect", self.disconnect)
        self.disconnect_btn.color = DesignSystem.COLORS['error']
        self.remove_drone_btn = Button(590, 200, 120, 40, "Remove", self.remove_drone)
        self.remove_drone_btn.color = DesignSystem.COLORS['error']
        
        # Drone list for selection
        self.drone_list_rect = pygame.Rect(200, 250, 600, 200)  # Will be drawn manually
        
        # Status display
        self.status_fields: Dict[str, Field] = {}
        self.status_data: Dict[str, Tuple[str, Tuple[int, int, int]]] = {}  # Store status data
        
        # Control tab components
        self.topic_list = TopicList(50, 100, 300, 500)
        self.topic_list.on_select = self.on_topic_selected
        self.control_topic_input = TextInput(370, 100, 380, 40, "/control")
        self.control_type_input = TextInput(770, 100, 380, 40, "controller_msgs/cmd")
        self.json_editor = JSONEditor(370, 160, 780, 340, '{\n    "cmd": 1\n}')
        self.command_history = []
        
        # Point cloud components - using professional renderer
        self.current_point_cloud = None
        self.pc_surface = None
        self.pc_surface_simple = None  # Simple rendering for status/pointcloud tabs
        self.pc_surface_o3d = None  # Open3D rendering surface
        self.pc_camera_angle_x = 0.0
        self.pc_camera_angle_y = 0.0
        self.pc_zoom = 1.0
        self.pc_last_render_time = 0.0
        self.pc_render_interval = 0.033  # ~30 FPS for point cloud updates
        self.pc_last_interaction_time = 0.0
        self.pc_interaction_throttle = 0.016  # ~60 FPS for mouse interactions
        
        # 3D view control buttons
        self.pc_view_controls = []  # Will be initialized in setup_ui
        
        # Initialize professional point cloud renderer
        if HAS_POINTCLOUD:
            self.pc_renderer = PointCloudRenderer(width=800, height=600)
        else:
            self.pc_renderer = None
        
        # Caching and threading for performance
        self.image_cache = None
        self.image_cache_lock = threading.Lock()
        self.pc_cache = None
        self.pc_cache_lock = threading.Lock()
        self.pc_render_cache = None  # Cached rendered surface
        self.pc_render_cache_params = None  # Cache parameters (angles, zoom)
        self.image_thread = None
        self.pc_thread = None
        self.render_thread = None
        self.stop_render_threads = threading.Event()
        
        # Open3D components (offscreen rendering)
        self.o3d_vis = None
        self.o3d_geometry = None
        self.o3d_window_created = False
        self.o3d_last_update = 0.0
        self.o3d_update_interval = 0.033  # ~30 FPS for Open3D updates
        self.o3d_window_size = (1200, 800)  # Fixed window size to prevent GUI scaling issues
        
        # Image update interval control
        self.image_last_update_time = 0.0
        self.image_update_interval = 0.033  # ~30 FPS for image updates
        
        # Network test components
        self.test_url_input = TextInput(200, 100, 400, 35, "ws://localhost:9090")
        self.test_timeout_input = TextInput(200, 150, 100, 35, "5")
        self.test_results = []
        self.test_btn = Button(200, 200, 150, 40, "Test Connection", self.test_connection)
        
        # Preset command buttons (fix lambda closure issue)
        preset_commands = [
            ("Takeoff", '{\n    "cmd": 1\n}'),
            ("Land", '{\n    "cmd": 2\n}'),
            ("Return", '{\n    "cmd": 3\n}'),
            ("Hover", '{\n    "cmd": 4\n}'),
        ]
        self.preset_buttons = []
        for i, (name, cmd) in enumerate(preset_commands):
            btn = Button(370 + i * 140, 520, 130, 38, name, 
                        lambda c=cmd: self.set_preset_command(c))
            self.preset_buttons.append(btn)
        self.format_json_btn = Button(370, 570, 140, 42, "Format JSON", 
                                     self.format_json, DesignSystem.COLORS['accent'])
        self.send_command_btn = Button(520, 570, 160, 42, "Send Command", 
                                            self.send_control_command,
                                      DesignSystem.COLORS['success'])
        
        # Professional display components
        # Status tab components (will be positioned in draw_status_tab)
        self.status_image_display = ImageDisplayComponent(0, 0, 0, 0, "Camera Stream")
        self.status_pointcloud_display = PointCloudDisplayComponent(0, 0, 0, 0, "Point Cloud")
        
        # Image tab component
        self.image_display = ImageDisplayComponent(0, 0, 0, 0, "Image Stream")
        
        # Point cloud tab component
        self.pointcloud_display = PointCloudDisplayComponent(0, 0, 0, 0, "Point Cloud Stream")
        
        # Card for point cloud tab (created in draw_pointcloud_tab, stored here for event handling)
        self.pc_card: Optional[Card] = None
        
        # Map component
        self.map_display = MapComponent(0, 0, 0, 0, "")
        
        # Card for map tab (created in draw_map_tab, stored here for event handling)
        self.map_card: Optional[Card] = None
        
    def on_topic_selected(self, topic_data: Tuple[str, str]):
        """Handle topic selection."""
        if isinstance(topic_data, tuple):
            self.control_topic_input.text = topic_data[0]
            if topic_data[1]:
                self.control_type_input.text = topic_data[1]
        else:
            self.control_topic_input.text = topic_data
        
    def setup_update_loop(self):
        """Setup periodic update loop with caching and threading at 30 FPS."""
        def update_loop():
            # Use 30 FPS update rate (0.033s interval)
            update_interval = 0.033
            while not self.stop_update.is_set():
                try:
                    # Update all connected drones
                    has_connected_drones = any(
                        drone_info.get('client') and drone_info.get('is_connected')
                        for drone_info in self.drones.values()
                    )
                    
                    if has_connected_drones:
                        # Update all connected drones
                        for drone_id, drone_info in self.drones.items():
                            if drone_info.get('client') and drone_info.get('is_connected'):
                                try:
                                    # Update trajectory for all drones
                                    pos = drone_info['client'].get_position()
                                    if len(pos) >= 3 and pos[0] != 0.0 and pos[1] != 0.0:
                                        drone_info['trajectory'].append((pos[0], pos[1], pos[2]))
                                        if len(drone_info['trajectory']) > 1000:
                                            drone_info['trajectory'] = drone_info['trajectory'][-1000:]
                                except:
                                    pass
                        
                        if self.current_tab == 1:  # Status tab
                            self.update_status()
                            if HAS_CV2:  # Update image in status tab
                                self.update_image_async()
                            if HAS_POINTCLOUD:  # Update point cloud in status tab
                                self.update_pointcloud_async()
                                # Sync camera from component
                                if self.status_pointcloud_display:
                                    angle_x, angle_y, zoom = self.status_pointcloud_display.get_camera()
                                    self.pc_camera_angle_x = angle_x
                                    self.pc_camera_angle_y = angle_y
                                    self.pc_zoom = zoom
                                # Trigger rendering after data update
                                self.update_pointcloud_simple()
                        if self.current_tab == 2 and HAS_CV2:  # Image tab
                            self.update_image_async()
                        if self.current_tab == 4 and HAS_POINTCLOUD:  # Point cloud tab
                            self.update_pointcloud_async()
                            # Sync camera from component
                            if self.pointcloud_display:
                                angle_x, angle_y, zoom = self.pointcloud_display.get_camera()
                                self.pc_camera_angle_x = angle_x
                                self.pc_camera_angle_y = angle_y
                                self.pc_zoom = zoom
                            # Trigger rendering after data update
                            self.update_pointcloud_simple()
                        if self.current_tab == 5 and HAS_OPEN3D:  # 3D View tab
                            self.update_pointcloud_open3d_async()
                        if self.current_tab == 3:  # Control tab
                            self.update_topic_list()
                except Exception as e:
                    print(f"Update error: {e}")
                time.sleep(update_interval)  # 30 FPS update rate
                
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
        
        # Start render threads for smooth interaction
        self.start_render_threads()
        
    def add_drone(self):
        """Add a new drone to the management list."""
        name = self.drone_name_input.text.strip()
        url = self.connection_url_input.text.strip()
        
        if not name:
            self.add_log("Error: Please enter drone name")
            return
        if not url:
            self.add_log("Error: Please enter WebSocket address")
            return
        
        # Generate unique ID
        drone_id = f"drone_{self.drone_counter}"
        self.drone_counter += 1
        
        # Add drone to dictionary
        self.drones[drone_id] = {
            'name': name,
            'url': url,
            'client': None,
            'is_connected': False,
            'use_mock': False,
            'image': None,
            'pointcloud': None,
            'trajectory': []  # List of (lat, lon, alt) tuples
        }
        
        # Set as current if first drone
        if self.current_drone_id is None:
            self.current_drone_id = drone_id
        
        self.add_log(f"Added drone: {name} ({url})")
    
    def remove_drone(self):
        """Remove the currently selected drone."""
        if self.current_drone_id is None:
            self.add_log("Error: No drone selected")
            return
        
        drone_id = self.current_drone_id
        drone_info = self.drones.get(drone_id)
        
        if drone_info:
            # Disconnect if connected
            if drone_info['client']:
                try:
                    drone_info['client'].terminate()
                except:
                    pass
            
            # Remove from dictionary
            del self.drones[drone_id]
            self.add_log(f"Removed drone: {drone_info['name']}")
            
            # Select another drone if available
            if len(self.drones) > 0:
                self.current_drone_id = list(self.drones.keys())[0]
            else:
                self.current_drone_id = None
                self.client = None
                self.is_connected = False
    
    def connect(self):
        """Connect the currently selected drone to ROS bridge."""
        if self.current_drone_id is None:
            self.add_log("Error: No drone selected")
            return
        
        drone_id = self.current_drone_id
        drone_info = self.drones[drone_id]
        url = drone_info['url']
        use_mock = self.use_mock_checkbox.checked
        
        def connect_thread():
            try:
                self.add_log(f"Connecting {drone_info['name']} to {url}...")
                
                if use_mock:
                    client = MockRosClient(url)
                    self.add_log(f"Using Mock Client (Test Mode) for {drone_info['name']}")
                else:
                    client = RosClient(url)
                    client.connect_async()
                    
                time.sleep(2)
                
                if client.is_connected():
                    drone_info['client'] = client
                    drone_info['is_connected'] = True
                    drone_info['use_mock'] = use_mock
                    
                    # Update legacy client for backward compatibility
                    self.client = client
                    self.is_connected = True
                    
                    self.add_log(f"Connection successful for {drone_info['name']}!")
                    self.update_topic_list()
                else:
                    self.add_log(f"Connection failed for {drone_info['name']}, please check address and network")
            except Exception as e:
                self.add_log(f"Connection error for {drone_info['name']}: {e}")
                
        threading.Thread(target=connect_thread, daemon=True).start()
        
    def disconnect(self):
        """Disconnect the currently selected drone from ROS bridge."""
        if self.current_drone_id is None:
            self.add_log("Error: No drone selected")
            return
        
        drone_id = self.current_drone_id
        drone_info = self.drones.get(drone_id)
        
        try:
            if drone_info and drone_info['client']:
                drone_info['client'].terminate()
                drone_info['client'] = None
                drone_info['is_connected'] = False
                self.add_log(f"Disconnected {drone_info['name']}")
            
            # Update legacy client
            if self.current_drone_id == drone_id:
                self.is_connected = False
                self.client = None
        except Exception as e:
            self.add_log(f"Disconnect error: {e}")
            
    def add_log(self, message: str):
        """Add log message."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        if not hasattr(self, 'connection_logs'):
            self.connection_logs = []
        self.connection_logs.append(log_entry)
        if len(self.connection_logs) > 50:
            self.connection_logs.pop(0)
            
    def update_status(self):
        """Update status display for the currently selected drone."""
        if self.current_drone_id is None:
            return
        
        drone_info = self.drones.get(self.current_drone_id)
        if not drone_info or not drone_info['client'] or not drone_info['is_connected']:
            return
        
        client = drone_info['client']
        # Update legacy client for backward compatibility
        self.client = client
        self.is_connected = drone_info['is_connected']
            
        try:
            state = client.get_status()
            pos = client.get_position()
            ori = client.get_orientation()
            
            # Update trajectory
            if len(pos) >= 3 and pos[0] != 0.0 and pos[1] != 0.0:  # Valid GPS coordinates
                drone_info['trajectory'].append((pos[0], pos[1], pos[2]))
                # Keep last 1000 points
                if len(drone_info['trajectory']) > 1000:
                    drone_info['trajectory'] = drone_info['trajectory'][-1000:]
            
            self.status_data = {
                "connected": ("Connected" if state.connected else "Disconnected",
                             DesignSystem.COLORS['success'] if state.connected else DesignSystem.COLORS['error']),
                "armed": ("Armed" if state.armed else "Disarmed",
                         DesignSystem.COLORS['warning'] if state.armed else DesignSystem.COLORS['text_secondary']),
                "mode": (state.mode or "N/A", DesignSystem.COLORS['text']),
                "battery": (f"{state.battery:.1f}%", 
                           DesignSystem.COLORS['error'] if state.battery < 20 else
                           DesignSystem.COLORS['warning'] if state.battery < 50 else
                           DesignSystem.COLORS['success']),
                "latitude": (f"{pos[0]:.6f}", DesignSystem.COLORS['text']),
                "longitude": (f"{pos[1]:.6f}", DesignSystem.COLORS['text']),
                "altitude": (f"{pos[2]:.2f}m", DesignSystem.COLORS['text']),
                "roll": (f"{ori[0]:.2f}°", DesignSystem.COLORS['text']),
                "pitch": (f"{ori[1]:.2f}°", DesignSystem.COLORS['text']),
                "yaw": (f"{ori[2]:.2f}°", DesignSystem.COLORS['text']),
                "landed": ("Landed" if state.landed else "Flying", DesignSystem.COLORS['text']),
                "reached": ("Yes" if state.reached else "No", DesignSystem.COLORS['text']),
                "returned": ("Yes" if state.returned else "No", DesignSystem.COLORS['text']),
                "tookoff": ("Yes" if state.tookoff else "No", DesignSystem.COLORS['text']),
            }
            
            # Update fields if they exist
            for key, (value, color) in self.status_data.items():
                if key in self.status_fields:
                    self.status_fields[key].set_value(value, color)
        except Exception as e:
            pass
            
    def update_image_async(self):
        """Update image display asynchronously with caching at 30 FPS."""
        if self.current_drone_id is None:
            return
        
        drone_info = self.drones.get(self.current_drone_id)
        if not drone_info or not drone_info.get('client') or not drone_info.get('is_connected'):
            return
        
        client = drone_info['client']
        # Update legacy client for backward compatibility
        self.client = client
        self.is_connected = drone_info['is_connected']
        
        # Throttle updates to 30 FPS
        current_time = time.time()
        if current_time - self.image_last_update_time < self.image_update_interval:
            return
        
        # Check if we need to update (non-blocking)
        if self.image_thread and self.image_thread.is_alive():
            return  # Already updating
            
        def update_image_thread():
            try:
                image_data = client.get_latest_image()
                if image_data:
                    frame, timestamp = image_data
                    max_width, max_height = 800, 600
                    h, w = frame.shape[:2]
                    scale = min(max_width / w, max_height / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame_resized = cv2.resize(frame, (new_w, new_h))
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    frame_rotated = np.rot90(frame_rgb, k=3)
                    frame_flipped = np.fliplr(frame_rotated)
                    new_image = pygame.surfarray.make_surface(frame_flipped)
                    
                    with self.image_cache_lock:
                        self.image_cache = new_image
                        self.current_image = new_image
                    self.image_last_update_time = time.time()
            except Exception:
                pass
        
        self.image_thread = threading.Thread(target=update_image_thread, daemon=True)
        self.image_thread.start()
    
    def update_image(self):
        """Update image display (synchronous fallback)."""
        self.update_image_async()
            
    def update_pointcloud_async(self):
        """Update point cloud display asynchronously with caching."""
        if self.current_drone_id is None:
            return
        
        drone_info = self.drones.get(self.current_drone_id)
        if not drone_info or not drone_info.get('client') or not drone_info.get('is_connected'):
            return
        
        client = drone_info['client']
        # Update legacy client for backward compatibility
        self.client = client
        self.is_connected = drone_info['is_connected']
        
        # Check if we need to update (non-blocking)
        if self.pc_thread and self.pc_thread.is_alive():
            return  # Already updating
            
        def update_pc_thread():
            try:
                pc_data = client.get_latest_point_cloud()
                if pc_data:
                    points, timestamp = pc_data
                    with self.pc_cache_lock:
                        self.pc_cache = points
                        self.current_point_cloud = points
                    # Invalidate cache to trigger re-render with new data
                    self.pc_render_cache_params = None
            except Exception as e:
                print(f"Point cloud update error: {e}")
        
        self.pc_thread = threading.Thread(target=update_pc_thread, daemon=True)
        self.pc_thread.start()
    
    def update_pointcloud_simple(self):
        """Update point cloud display with simple rendering (for status/pointcloud tabs)."""
        # Check if we have point cloud data
        if self.current_point_cloud is None:
            return
        
        current_time = time.time()
        # Always render if enough time has passed or cache is invalid
        if current_time - self.pc_last_render_time >= self.pc_render_interval:
            # Check if cache is valid
            cache_key = (self.pc_camera_angle_x, self.pc_camera_angle_y, self.pc_zoom)
            if self.pc_render_cache_params != cache_key or self.pc_render_cache is None:
                # Need to re-render
                self.render_pointcloud_simple()
                self.pc_render_cache_params = cache_key
                self.pc_last_render_time = current_time
            else:
                # Use cached render
                with self.pc_cache_lock:
                    if self.pc_render_cache is not None:
                        self.pc_surface_simple = self.pc_render_cache
    
    def update_pointcloud_open3d(self):
        """Update point cloud display using Open3D (for 3D View tab)."""
        if self.current_drone_id is None:
            return
        
        drone_info = self.drones.get(self.current_drone_id)
        if not drone_info or not drone_info.get('client') or not drone_info.get('is_connected'):
            return
        
        client = drone_info['client']
        # Update legacy client for backward compatibility
        self.client = client
        self.is_connected = drone_info['is_connected']
            
        try:
            pc_data = client.get_latest_point_cloud()
            if pc_data:
                points, timestamp = pc_data
                self.current_point_cloud = points
                self.render_pointcloud_open3d()
        except Exception:
            pass
            
    def start_render_threads(self):
        """Start background render threads for smooth interaction."""
        def render_loop():
            while not self.stop_render_threads.is_set():
                try:
                    if HAS_POINTCLOUD and self.pc_renderer is not None:
                        with self.pc_cache_lock:
                            has_data = self.current_point_cloud is not None
                        if has_data:
                            current_time = time.time()
                            if current_time - self.pc_last_render_time >= self.pc_render_interval:
                                # Check if cache is valid
                                cache_key = (self.pc_camera_angle_x, self.pc_camera_angle_y, self.pc_zoom)
                                with self.pc_cache_lock:
                                    cache_valid = (self.pc_render_cache_params == cache_key and 
                                                 self.pc_render_cache is not None)
                                if not cache_valid:
                                    # Render in background using professional renderer
                                    self.render_pointcloud_simple_background()
                                    with self.pc_cache_lock:
                                        self.pc_render_cache_params = cache_key
                                    self.pc_last_render_time = current_time
                except Exception as e:
                    print(f"Render thread error: {e}")
                time.sleep(0.01)  # Small sleep to avoid busy waiting
        
        self.render_thread = threading.Thread(target=render_loop, daemon=True)
        self.render_thread.start()
    
    def render_pointcloud_simple(self):
        """Render point cloud using simple 3D projection (synchronous)."""
        if not HAS_POINTCLOUD or self.current_point_cloud is None:
            return
        self.render_pointcloud_simple_background()
    
    def render_pointcloud_simple_background(self):
        """Render point cloud in background thread using professional renderer."""
        if not HAS_POINTCLOUD or self.pc_renderer is None:
            return
            
        try:
            with self.pc_cache_lock:
                points = self.current_point_cloud
            if points is None:
                return
            
            # Update renderer camera
            self.pc_renderer.set_camera(self.pc_camera_angle_x, 
                                       self.pc_camera_angle_y, 
                                       self.pc_zoom)
            
            # Render using professional renderer
            surface = self.pc_renderer.render(points)
            
            if surface is not None:
                # Update cache and surface
                with self.pc_cache_lock:
                    self.pc_render_cache = surface
                    self.pc_surface_simple = surface
            
        except Exception as e:
            print(f"Point cloud simple render error: {e}")
            import traceback
            traceback.print_exc()
    
    def update_pointcloud_open3d_async(self):
        """Update point cloud using Open3D asynchronously."""
        if not HAS_OPEN3D or self.current_point_cloud is None:
            return
        self.render_pointcloud_open3d()
    
    def render_pointcloud_open3d(self):
        """Render point cloud using Open3D offscreen rendering (for 3D View tab)."""
        if not HAS_OPEN3D or self.current_point_cloud is None:
            return
            
        try:
            import numpy as np
            
            current_time = time.time()
            # Throttle updates for performance
            if current_time - self.o3d_last_update < self.o3d_update_interval:
                return
            
            with self.pc_cache_lock:
                points = self.current_point_cloud
            if points is None or len(points) == 0:
                return
            
            # Convert to numpy array
            if not isinstance(points, np.ndarray):
                points = np.array(points, dtype=np.float32)
            
            # Sample points if too many for performance (reduced for better performance)
            max_points = 50000  # Reduced from 100000 for better performance
            if len(points) > max_points:
                step = len(points) // max_points
                points = points[::step]
            
            # Create or update Open3D point cloud
            if self.o3d_geometry is None:
                self.o3d_geometry = o3d.geometry.PointCloud()
            
            # Update point cloud data
            self.o3d_geometry.points = o3d.utility.Vector3dVector(points)
            
            # Color points based on Z coordinate (white gradient for black theme)
            if len(points) > 0:
                z_coords = points[:, 2]
                z_min, z_max = np.min(z_coords), np.max(z_coords)
                if z_max > z_min:
                    z_normalized = (z_coords - z_min) / (z_max - z_min)
                    # Use white gradient (black theme compatible)
                    colors = np.zeros((len(points), 3))
                    colors[:, 0] = z_normalized  # R
                    colors[:, 1] = z_normalized   # G
                    colors[:, 2] = z_normalized  # B (white gradient)
                    self.o3d_geometry.colors = o3d.utility.Vector3dVector(colors)
                else:
                    # Single color - white
                    self.o3d_geometry.paint_uniform_color([1.0, 1.0, 1.0])
            
            # Create or update offscreen visualizer with fixed size
            if not self.o3d_window_created:
                self.o3d_vis = o3d.visualization.Visualizer()
                # Use fixed window size to prevent GUI scaling issues
                self.o3d_vis.create_window("3D Point Cloud View", 
                                        width=self.o3d_window_size[0], 
                                        height=self.o3d_window_size[1], 
                                        visible=False)
                self.o3d_vis.add_geometry(self.o3d_geometry)
                
                # Set black background
                opt = self.o3d_vis.get_render_option()
                opt.background_color = np.array([0.0, 0.0, 0.0])
                opt.point_size = 1.5  # Reduced for better performance
                
                # Set view point
                ctr = self.o3d_vis.get_view_control()
                ctr.set_zoom(0.7)
                
                self.o3d_window_created = True
            else:
                self.o3d_vis.update_geometry(self.o3d_geometry)
            
            # Render to image
            self.o3d_vis.poll_events()
            self.o3d_vis.update_renderer()
            
            # Capture rendered image
            img = self.o3d_vis.capture_screen_float_buffer(do_render=True)
            img_array = np.asarray(img)
            img_array = (img_array * 255).astype(np.uint8)
            img_array = np.flipud(img_array)  # Flip vertically for pygame
            
            # Convert to pygame surface
            self.pc_surface_o3d = pygame.surfarray.make_surface(img_array.swapaxes(0, 1))
            
            self.o3d_last_update = current_time
            
        except Exception as e:
            print(f"Open3D render error: {e}")
    
    def render_pointcloud(self):
        """Render point cloud using optimized vectorized 3D projection."""
        if not HAS_POINTCLOUD or self.current_point_cloud is None:
            return
            
        try:
            points = self.current_point_cloud
            if len(points) == 0:
                return
            
            # Ensure numpy is available (required for performance)
            try:
                import numpy as np
            except ImportError:
                print("Warning: numpy required for optimized point cloud rendering")
                return
            
            # Convert to numpy array if not already
            if not isinstance(points, np.ndarray):
                points = np.array(points, dtype=np.float32)
                
            # Adaptive sampling based on point count for smooth performance
            max_points = 30000  # Reduced for better performance
            if len(points) > max_points:
                step = len(points) // max_points
                points = points[::step]
            
            # Create surface for rendering
            width, height = 800, 600
            self.pc_surface = pygame.Surface((width, height))
            self.pc_surface.fill(DesignSystem.COLORS['bg'])
            
            # Vectorized center calculation
            center = np.mean(points, axis=0)
            
            # Vectorized distance calculation
            points_centered = points - center
            distances = np.linalg.norm(points_centered, axis=1)
            max_dist = np.max(distances) if len(distances) > 0 else 1.0
            if max_dist == 0:
                max_dist = 1.0
            
            scale = min(width, height) * 0.4 / max_dist * self.pc_zoom
            
            # Pre-compute rotation matrices
            cos_x, sin_x = math.cos(self.pc_camera_angle_x), math.sin(self.pc_camera_angle_x)
            cos_y, sin_y = math.cos(self.pc_camera_angle_y), math.sin(self.pc_camera_angle_y)
            
            # Vectorized rotation and projection
            x, y, z = points_centered[:, 0], points_centered[:, 1], points_centered[:, 2]
            
            # Rotate around X axis (vectorized)
            y_rot = y * cos_x - z * sin_x
            z_rot = y * sin_x + z * cos_x
            
            # Rotate around Y axis (vectorized)
            x_final = x * cos_y + z_rot * sin_y
            z_final = -x * sin_y + z_rot * cos_y
            
            # Filter points in front of camera
            front_mask = z_final > -max_dist * 0.1
            x_final = x_final[front_mask]
            y_final = y_rot[front_mask]
            z_final = z_final[front_mask]
            
            if len(x_final) == 0:
                return
            
            # Vectorized perspective projection
            z_scale = 1.0 + z_final / max_dist
            proj_x = (width / 2 + x_final * scale / z_scale).astype(np.int32)
            proj_y = (height / 2 - y_final * scale / z_scale).astype(np.int32)
            
            # Filter points within screen bounds (with extra margin for 2x2 pixel drawing)
            valid_mask = (proj_x >= 0) & (proj_x < width - 1) & (proj_y >= 0) & (proj_y < height - 1)
            proj_x = proj_x[valid_mask]
            proj_y = proj_y[valid_mask]
            z_final = z_final[valid_mask]
            
            if len(proj_x) == 0:
                return
            
            # Vectorized color calculation based on depth
            z_normalized = np.clip((z_final + max_dist) / (2 * max_dist), 0.0, 1.0)
            primary_r, primary_g, primary_b = DesignSystem.COLORS['primary']
            colors_r = (primary_r * z_normalized).astype(np.uint8)
            colors_g = (primary_g * z_normalized).astype(np.uint8)
            colors_b = np.full(len(z_normalized), primary_b, dtype=np.uint8)
            
            # Use pixel array for fast batch drawing (note: pygame uses [y, x] indexing)
            pixel_array_3d = pygame.surfarray.pixels3d(self.pc_surface)
            
            # Batch draw points using array indexing (much faster than individual draws)
            # pygame.surfarray uses [y, x] indexing, so we need to swap
            # Clip coordinates to ensure they're within bounds (with margin for 2x2 pixels)
            proj_x_clipped = np.clip(proj_x, 0, width - 2).astype(np.int32)
            proj_y_clipped = np.clip(proj_y, 0, height - 2).astype(np.int32)
            
            for i in range(len(proj_x_clipped)):
                px, py = int(proj_x_clipped[i]), int(proj_y_clipped[i])
                # Ensure bounds (with margin for 2x2 drawing)
                if 0 <= px < width - 1 and 0 <= py < height - 1:
                    color = (int(colors_r[i]), int(colors_g[i]), int(colors_b[i]))
                    # Draw 2x2 pixel block (all within bounds)
                    pixel_array_3d[py, px] = color
                    pixel_array_3d[py, px + 1] = color
                    pixel_array_3d[py + 1, px] = color
                    pixel_array_3d[py + 1, px + 1] = color
            
            del pixel_array_3d  # Unlock surface
            
            # Draw axes (optimized)
            axis_length = max_dist * 0.3
            origin_x, origin_y = width // 2, height // 2
            
            # Pre-compute axis endpoints
            axis_points = np.array([
                [axis_length, 0, 0],  # X axis
                [0, axis_length, 0],  # Y axis
                [0, 0, axis_length],  # Z axis
            ])
            
            axis_colors = [
                DesignSystem.COLORS['error'],    # X - red
                DesignSystem.COLORS['success'],  # Y - green
                DesignSystem.COLORS['primary'],  # Z - blue
            ]
            
            for axis_point, color in zip(axis_points, axis_colors):
                axis_2d = self.project_point_3d(axis_point, center, max_dist, scale, width, height)
                if axis_2d:
                    pygame.draw.line(self.pc_surface, color, 
                                   (origin_x, origin_y), axis_2d, 2)
            
            # Draw info text (if renderer stats available, use them)
            font = DesignSystem.get_font('small')
            if self.pc_renderer and hasattr(self.pc_renderer, 'render_stats'):
                stats = self.pc_renderer.render_stats
                info_text = (f"Points: {stats.get('total_points', len(points))} → "
                           f"{stats.get('rendered_points', len(proj_x))} | "
                           f"Zoom: {self.pc_zoom:.2f} | "
                           f"FPS: {stats.get('fps', 0):.1f}")
            else:
                info_text = f"Points: {len(points)} | Rendered: {len(proj_x)} | Zoom: {self.pc_zoom:.2f}"
            text_surf = font.render(info_text, True, DesignSystem.COLORS['text_secondary'])
            self.pc_surface.blit(text_surf, (10, 10))
            
        except Exception as e:
            print(f"Point cloud render error: {e}")
            
    def project_point_3d(self, point, center, max_dist, scale, width, height):
        """Project a 3D point to 2D screen coordinates."""
        try:
            x, y, z = point[0] - center[0], point[1] - center[1], point[2] - center[2]
            
            cos_x, sin_x = math.cos(self.pc_camera_angle_x), math.sin(self.pc_camera_angle_x)
            cos_y, sin_y = math.cos(self.pc_camera_angle_y), math.sin(self.pc_camera_angle_y)
            
            y, z = y * cos_x - z * sin_x, y * sin_x + z * cos_x
            x, z = x * cos_y + z * sin_y, -x * sin_y + z * cos_y
            
            if z > -max_dist * 0.1:
                if z != 0:
                    proj_x = int(width / 2 + x * scale / (1 + z / max_dist))
                    proj_y = int(height / 2 - y * scale / (1 + z / max_dist))
                else:
                    proj_x = int(width / 2 + x * scale)
                    proj_y = int(height / 2 - y * scale)
                
                if 0 <= proj_x < width and 0 <= proj_y < height:
                    return (proj_x, proj_y)
        except:
            pass
        return None
            
    def reset_pc_view(self):
        """Reset point cloud view to default."""
        self.pc_camera_angle_x = 0.0
        self.pc_camera_angle_y = 0.0
        self.pc_zoom = 1.0
        self.pc_render_cache_params = None  # Invalidate cache
    
    def set_pc_top_view(self):
        """Set point cloud to top view (looking down from +Z)."""
        self.pc_camera_angle_x = -math.pi / 2  # Look down
        self.pc_camera_angle_y = 0.0
        self.pc_zoom = 1.0
        self.pc_render_cache_params = None
    
    def set_pc_front_view(self):
        """Set point cloud to front view (looking from +Y)."""
        self.pc_camera_angle_x = 0.0
        self.pc_camera_angle_y = 0.0
        self.pc_zoom = 1.0
        self.pc_render_cache_params = None
    
    def set_pc_side_view(self):
        """Set point cloud to side view (looking from +X)."""
        self.pc_camera_angle_x = 0.0
        self.pc_camera_angle_y = math.pi / 2  # Rotate 90 degrees
        self.pc_zoom = 1.0
        self.pc_render_cache_params = None
    
    def set_pc_iso_view(self):
        """Set point cloud to isometric view."""
        self.pc_camera_angle_x = -math.pi / 6  # 30 degrees down
        self.pc_camera_angle_y = math.pi / 4  # 45 degrees rotated
        self.pc_zoom = 1.0
        self.pc_render_cache_params = None
    
    def set_preset_command(self, command):
        """Set preset command."""
        # Handle both string and dict (from port system)
        if isinstance(command, dict):
            # If called from port system with dict, extract command from button text or use default
            # This shouldn't happen with lambda, but handle it gracefully
            command = command.get('text', '')
            # Try to find the command from button text
            preset_commands = {
                'Takeoff': '{\n    "cmd": 1\n}',
                'Land': '{\n    "cmd": 2\n}',
                'Return': '{\n    "cmd": 3\n}',
                'Hover': '{\n    "cmd": 4\n}',
            }
            command = preset_commands.get(command, '')
        
        # Ensure command is a string
        if not isinstance(command, str):
            command = str(command)
            
        self.json_editor.text = command
        self.json_editor.cursor_pos = [0, 0]
        
    def format_json(self):
        """Format JSON in editor."""
        self.json_editor.format_json()
        
    def update_topic_list(self):
        """Update topic list from currently selected drone."""
        try:
            from rosclient.clients.config import DEFAULT_TOPICS
            topics = [(topic.name, topic.type) for topic in DEFAULT_TOPICS.values()]
            
            if self.current_drone_id is not None:
                drone_info = self.drones.get(self.current_drone_id)
                if drone_info and drone_info.get('client') and drone_info.get('is_connected'):
                    client = drone_info['client']
                    # Update legacy client for backward compatibility
                    self.client = client
                    self.is_connected = drone_info['is_connected']
                    
                    if hasattr(client, '_ts_mgr') and client._ts_mgr:
                        if hasattr(client._ts_mgr, '_topics'):
                            for topic_name in client._ts_mgr._topics.keys():
                                if not any(t[0] == topic_name for t in topics):
                                    topics.append((topic_name, ""))
            
            topics = sorted(topics, key=lambda x: x[0])
            self.topic_list.set_items(topics)
        except Exception as e:
            self.topic_list.set_items([])
        
    def send_control_command(self):
        """Send control command to currently selected drone."""
        if self.current_drone_id is None:
            self.add_log("Warning: No drone selected")
            return
        
        drone_info = self.drones.get(self.current_drone_id)
        if not drone_info or not drone_info.get('client') or not drone_info.get('is_connected'):
            self.add_log("Warning: Selected drone is not connected")
            return
        
        client = drone_info['client']
        # Update legacy client for backward compatibility
        self.client = client
        self.is_connected = drone_info['is_connected']
            
        try:
            topic = self.control_topic_input.text.strip()
            topic_type = self.control_type_input.text.strip()
            # Ensure text is a string, not a dict
            editor_text = self.json_editor.text
            if isinstance(editor_text, dict):
                # If somehow text became a dict, try to extract or use empty
                message_text = ""
            else:
                message_text = str(editor_text).strip()
            
            if not topic or not message_text:
                self.add_log("Warning: Please fill in Topic and message content")
                return
                
            try:
                message = json.loads(message_text)
            except json.JSONDecodeError as e:
                self.add_log(f"Error: JSON format error: {e}")
                return
                
            client.publish(topic, topic_type, message)
            
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
        url = self.test_url_input.text.strip()
        timeout = float(self.test_timeout_input.text or "5")
        
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
        
    def draw_tab_bar(self):
        """Draw tab navigation bar."""
        tab_height = LayoutManager.TAB_BAR_HEIGHT
        tab_width = self.screen_width // len(self.tabs)
        
        # Draw background
        tab_bar_rect = pygame.Rect(0, 0, self.screen_width, tab_height)
        pygame.draw.rect(self.screen, DesignSystem.COLORS['bg_secondary'], tab_bar_rect)
        pygame.draw.line(self.screen, DesignSystem.COLORS['border'],
                        (0, tab_bar_rect.bottom - 1),
                        (self.screen_width, tab_bar_rect.bottom - 1), 2)
        
        # Draw tabs
        font = DesignSystem.get_font('label')
        for i, tab_name in enumerate(self.tabs):
            tab_rect = pygame.Rect(i * tab_width, 0, tab_width, tab_height)
            
            if i == self.current_tab:
                # Active tab
                active_rect = pygame.Rect(tab_rect.x + 2, tab_rect.y + 2,
                                        tab_rect.width - 4, tab_rect.height - 2)
                pygame.draw.rect(self.screen, DesignSystem.COLORS['surface'], active_rect,
                               border_radius=DesignSystem.RADIUS['sm'])
                pygame.draw.line(self.screen, DesignSystem.COLORS['primary'],
                               (tab_rect.x, tab_rect.bottom - 2),
                               (tab_rect.right, tab_rect.bottom - 2), 3)
                text_color = DesignSystem.COLORS['text']
            else:
                text_color = DesignSystem.COLORS['text_secondary']
            
            text_surf = font.render(tab_name, True, text_color)
            text_rect = text_surf.get_rect(center=tab_rect.center)
            self.screen.blit(text_surf, text_rect)
        
    def draw_connection_tab(self):
        """Draw connection configuration tab with multi-drone management."""
        # Tab bar height is 45, add padding below it
        tab_height = 45
        y = tab_height + DesignSystem.SPACING['lg']
        
        # Title
        title_label = Label(50, y, "Multi-Drone Connection Management", 'title', 
                           DesignSystem.COLORS['text'])
        title_label.draw(self.screen)
        y += 50
        
        # Connection settings card
        settings_card = Card(50, y, self.screen_width - 100, 280, "Add/Connect Drone")
        settings_card.draw(self.screen)
        
        card_y = y + 50
        # Drone name
        name_label = Label(70, card_y, "Drone Name:", 'label',
                         DesignSystem.COLORS['text_label'])
        name_label.draw(self.screen)
        self.drone_name_input.rect.x = 70
        self.drone_name_input.rect.y = card_y + 25
        self.drone_name_input.rect.width = 200
        self.drone_name_input.draw(self.screen)
        
        # Connection URL
        url_label = Label(290, card_y, "WebSocket Address:", 'label',
                         DesignSystem.COLORS['text_label'])
        url_label.draw(self.screen)
        self.connection_url_input.rect.x = 290
        self.connection_url_input.rect.y = card_y + 25
        self.connection_url_input.rect.width = settings_card.rect.width - 310
        self.connection_url_input.draw(self.screen)
        card_y += 70
        
        # Mock checkbox
        self.use_mock_checkbox.rect.x = 70
        self.use_mock_checkbox.rect.y = card_y
        self.use_mock_checkbox.draw(self.screen)
        card_y += 50
        
        # Buttons
        self.add_drone_btn.rect.x = 70
        self.add_drone_btn.rect.y = card_y
        self.add_drone_btn.draw(self.screen)
        self.connect_btn.rect.x = 200
        self.connect_btn.rect.y = card_y
        self.connect_btn.draw(self.screen)
        self.disconnect_btn.rect.x = 330
        self.disconnect_btn.rect.y = card_y
        self.disconnect_btn.draw(self.screen)
        self.remove_drone_btn.rect.x = 460
        self.remove_drone_btn.rect.y = card_y
        self.remove_drone_btn.draw(self.screen)
        
        y += settings_card.rect.height + 20
        
        # Drone list card
        list_card = Card(50, y, self.screen_width - 100, 200, "Connected Drones")
        list_card.draw(self.screen)
        
        list_area = pygame.Rect(70, y + 50, list_card.rect.width - 40, list_card.rect.height - 70)
        pygame.draw.rect(self.screen, DesignSystem.COLORS['bg'], list_area,
                       border_radius=DesignSystem.RADIUS['sm'])
        
        # Draw drone list
        font = DesignSystem.get_font('label')
        list_y = list_area.y + DesignSystem.SPACING['sm']
        for i, (drone_id, drone_info) in enumerate(self.drones.items()):
            if list_y + 30 > list_area.bottom - DesignSystem.SPACING['sm']:
                break
            
            # Highlight selected drone
            if drone_id == self.current_drone_id:
                highlight_rect = pygame.Rect(list_area.x + 5, list_y - 5, list_area.width - 10, 30)
                pygame.draw.rect(self.screen, DesignSystem.COLORS['primary'], highlight_rect,
                               border_radius=DesignSystem.RADIUS['sm'])
            
            # Drone info
            status = "Connected" if drone_info.get('is_connected') else "Disconnected"
            status_color = DesignSystem.COLORS['success'] if drone_info.get('is_connected') else DesignSystem.COLORS['error']
            text = f"{drone_info['name']} - {drone_info['url']} [{status}]"
            text_surf = font.render(text, True, DesignSystem.COLORS['text'])
            self.screen.blit(text_surf, (list_area.x + DesignSystem.SPACING['md'], list_y))
            list_y += 35
        
        if len(self.drones) == 0:
            empty_text = font.render("No drones added. Add a drone above.", True, 
                                   DesignSystem.COLORS['text_secondary'])
            self.screen.blit(empty_text, (list_area.x + DesignSystem.SPACING['md'], list_y))
        
        y += list_card.rect.height + 20
        
        # Log display card
        log_card = Card(50, y, self.screen_width - 100, 
                       self.screen_height - y - 20, "Connection Log")
        log_card.draw(self.screen)
        
        log_area = pygame.Rect(70, y + 50, log_card.rect.width - 40, 
                              log_card.rect.height - 70)
        pygame.draw.rect(self.screen, DesignSystem.COLORS['bg'], log_area,
                       border_radius=DesignSystem.RADIUS['sm'])
        
        if hasattr(self, 'connection_logs'):
            font = DesignSystem.get_font('console')
            log_y = log_area.y + DesignSystem.SPACING['sm']
            for log in self.connection_logs[-20:]:
                log_surf = font.render(log, True, DesignSystem.COLORS['text_console'])
                if log_y + log_surf.get_height() > log_area.bottom - DesignSystem.SPACING['sm']:
                    break
                self.screen.blit(log_surf, (log_area.x + DesignSystem.SPACING['md'], log_y))
                log_y += log_surf.get_height() + DesignSystem.SPACING['xs']
                    
    def draw_status_tab(self):
        """Draw status monitoring tab with camera and point cloud."""
        # Tab bar height is 45, add padding below it
        tab_height = 45
        y = tab_height + DesignSystem.SPACING['lg']
        
        # Title with connection indicator
        title_label = Label(50, y, "Status Monitoring", 'title',
                           DesignSystem.COLORS['text'])
        title_label.draw(self.screen)
        
        # Connection indicator - show status of currently selected drone
        current_connected = False
        if self.current_drone_id:
            drone_info = self.drones.get(self.current_drone_id)
            if drone_info:
                current_connected = drone_info.get('is_connected', False)
        
        indicator_color = (DesignSystem.COLORS['success'] if current_connected 
                          else DesignSystem.COLORS['error'])
        indicator_text = "Connected" if current_connected else "Disconnected"
        indicator_label = Label(self.screen_width - 200, y, indicator_text, 'label', indicator_color)
        indicator_label.draw(self.screen)
        y += 50
        
        # Status fields (top section)
        status_fields_data = [
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
        
        x_start = 50
        x_offset = (self.screen_width - 100) // 2
        card_width = x_offset - DesignSystem.SPACING['md']
        card_height = 40
        
        # Draw status fields in compact layout (2 columns, 7 rows)
        status_y = y
        for i, (label, field_key) in enumerate(status_fields_data):
            x = x_start if i % 2 == 0 else x_start + x_offset
            row = i // 2
            card_y = status_y + row * (card_height + DesignSystem.SPACING['sm'])
            
            if field_key not in self.status_fields:
                value, color = self.get_status_value(field_key)
                self.status_fields[field_key] = Field(x, card_y, card_width, card_height, label, value, color)
            else:
                value, color = self.get_status_value(field_key)
                self.status_fields[field_key].set_value(value, color)
            
            self.status_fields[field_key].draw(self.screen)
        
        # Calculate bottom of status fields
        status_bottom = status_y + ((len(status_fields_data) + 1) // 2) * (card_height + DesignSystem.SPACING['sm'])
        y = status_bottom + DesignSystem.SPACING['xl']
        
        # Camera and Point Cloud display (side by side) - using professional components with Cards
        display_height = self.screen_height - y - 20
        gap = DesignSystem.SPACING['lg']
        total_width = self.screen_width - 100
        display_width = (total_width - gap) // 2
        
        # Camera display (left) - using Card and ImageDisplayComponent
        if HAS_CV2:
            camera_card = Card(50, y, display_width, display_height, "Camera Feed")
            
            # Get content area from card (below title)
            content_area = camera_card.get_content_area()
            
            # Image display area (inside card's content area, with proper padding)
            img_padding = DesignSystem.SPACING['md']
            # Coordinates are relative to card's content area
            self.status_image_display.rect.x = img_padding
            self.status_image_display.rect.y = img_padding
            self.status_image_display.rect.width = content_area.width - img_padding * 2
            self.status_image_display.rect.height = content_area.height - img_padding * 2
            self.status_image_display.set_image(self.current_image)
            
            # Add component to card as child
            camera_card.add_child(self.status_image_display)
            
            # Draw card (which will draw children in correct area)
            camera_card.draw(self.screen)
        
        # Point Cloud display (right) - using Card and PointCloudDisplayComponent
        if HAS_POINTCLOUD:
            pc_x = 50 + display_width + gap
            # Store card reference for event handling
            if not hasattr(self, 'status_pc_card') or self.status_pc_card is None:
                self.status_pc_card = Card(pc_x, y, display_width, display_height, "Point Cloud")
            else:
                # Update card position and size
                self.status_pc_card.rect.x = pc_x
                self.status_pc_card.rect.y = y
                self.status_pc_card.rect.width = display_width
                self.status_pc_card.rect.height = display_height
                # Clear children and re-add
                self.status_pc_card.children.clear()
            
            # Get content area from card (below title)
            content_area = self.status_pc_card.get_content_area()
            
            # Point cloud display area (inside card's content area, with proper padding)
            pc_padding = DesignSystem.SPACING['md']
            # Coordinates are relative to card's content area
            self.status_pointcloud_display.rect.x = pc_padding
            self.status_pointcloud_display.rect.y = pc_padding
            self.status_pointcloud_display.rect.width = content_area.width - pc_padding * 2
            self.status_pointcloud_display.rect.height = content_area.height - pc_padding * 2
            # Don't show title in component since Card already has one
            self.status_pointcloud_display.title = ""
            self.status_pointcloud_display.set_pointcloud(self.pc_surface_simple)
            self.status_pointcloud_display.set_camera(self.pc_camera_angle_x, 
                                                     self.pc_camera_angle_y, 
                                                     self.pc_zoom)
            
            # Add component to card as child
            self.status_pc_card.add_child(self.status_pointcloud_display)
            
            # Draw card (which will draw children in correct area)
            self.status_pc_card.draw(self.screen)
            
    def get_status_value(self, field_key: str) -> Tuple[str, Tuple[int, int, int]]:
        """Get status value and color for a field."""
        # Use stored status data if available
        if field_key in self.status_data:
            return self.status_data[field_key]
        # Return defaults if not available
        defaults = {
            "connected": ("N/A", DesignSystem.COLORS['text_tertiary']),
            "armed": ("N/A", DesignSystem.COLORS['text_tertiary']),
            "mode": ("N/A", DesignSystem.COLORS['text_tertiary']),
            "battery": ("N/A", DesignSystem.COLORS['text_tertiary']),
            "latitude": ("N/A", DesignSystem.COLORS['text_tertiary']),
            "longitude": ("N/A", DesignSystem.COLORS['text_tertiary']),
            "altitude": ("N/A", DesignSystem.COLORS['text_tertiary']),
            "roll": ("N/A", DesignSystem.COLORS['text_tertiary']),
            "pitch": ("N/A", DesignSystem.COLORS['text_tertiary']),
            "yaw": ("N/A", DesignSystem.COLORS['text_tertiary']),
            "landed": ("N/A", DesignSystem.COLORS['text_tertiary']),
            "reached": ("N/A", DesignSystem.COLORS['text_tertiary']),
            "returned": ("N/A", DesignSystem.COLORS['text_tertiary']),
            "tookoff": ("N/A", DesignSystem.COLORS['text_tertiary']),
        }
        return defaults.get(field_key, ("N/A", DesignSystem.COLORS['text_tertiary']))
            
    def draw_image_tab(self):
        """Draw image display tab using professional component."""
        # Tab bar height is 45, add padding below it
        tab_height = 45
        y = tab_height + DesignSystem.SPACING['lg']
        
        title_label = Label(50, y, "Image Display", 'title', DesignSystem.COLORS['text'])
        title_label.draw(self.screen)
        y += 50
        
        # Use professional ImageDisplayComponent
        self.image_display.rect.x = 50
        self.image_display.rect.y = y
        self.image_display.rect.width = self.screen_width - 100
        self.image_display.rect.height = self.screen_height - y - 20
        self.image_display.set_image(self.current_image)
        self.image_display.draw(self.screen)
            
    def draw_control_tab(self):
        """Draw control command tab."""
        # Tab bar height is 45, add padding below it
        tab_height = 45
        y = tab_height + DesignSystem.SPACING['lg']
        
        title_label = Label(50, y, "Control Commands", 'title', DesignSystem.COLORS['text'])
        title_label.draw(self.screen)
        y += 50
        
        # Left panel: Topic list
        topic_card = Card(50, y, 320, 550, "Available Topics")
        topic_card.draw(self.screen)
        
        self.topic_list.rect.x = 70
        self.topic_list.rect.y = y + 50
        self.topic_list.rect.width = topic_card.rect.width - 40
        self.topic_list.rect.height = topic_card.rect.height - 70
        self.topic_list.draw(self.screen)
        
        # Right panel
        x_right = 50 + topic_card.rect.width + 20
        y_right = y
        
        # Topic configuration
        config_card = Card(x_right, y_right, self.screen_width - x_right - 50, 120, "Topic Configuration")
        config_card.draw(self.screen)
        
        card_y = y_right + 50
        topic_label = Label(x_right + 20, card_y, "Topic Name:", 'label',
                           DesignSystem.COLORS['text_label'])
        topic_label.draw(self.screen)
        self.control_topic_input.rect.x = x_right + 20
        self.control_topic_input.rect.y = card_y + 25
        self.control_topic_input.rect.width = (config_card.rect.width - 60) // 2
        self.control_topic_input.draw(self.screen)
        
        type_label = Label(x_right + 20 + self.control_topic_input.rect.width + 20, card_y,
                          "Topic Type:", 'label', DesignSystem.COLORS['text_label'])
        type_label.draw(self.screen)
        self.control_type_input.rect.x = x_right + 20 + self.control_topic_input.rect.width + 20
        self.control_type_input.rect.y = card_y + 25
        self.control_type_input.rect.width = config_card.rect.width - self.control_topic_input.rect.width - 80
        self.control_type_input.draw(self.screen)
        
        y_right += config_card.rect.height + 20
        
        # JSON editor
        editor_card = Card(x_right, y_right, self.screen_width - x_right - 50, 380, "Message Content (JSON)")
        editor_card.draw(self.screen)
        
        self.json_editor.rect.x = x_right + 20
        self.json_editor.rect.y = y_right + 50
        self.json_editor.rect.width = editor_card.rect.width - 40
        self.json_editor.rect.height = editor_card.rect.height - 70
        self.json_editor.draw(self.screen)
        
        y_right += editor_card.rect.height + 20
        
        # Action buttons
        button_y = y_right
        button_x = x_right + 20
        
        for i, btn in enumerate(self.preset_buttons):
            btn.rect.x = button_x + i * 140
            btn.rect.y = button_y
            btn.draw(self.screen)
        
        self.format_json_btn.rect.x = button_x + len(self.preset_buttons) * 140
        self.format_json_btn.rect.y = button_y
        self.format_json_btn.draw(self.screen)
        
        self.send_command_btn.rect.x = button_x + len(self.preset_buttons) * 140 + 150
        self.send_command_btn.rect.y = button_y
        self.send_command_btn.draw(self.screen)
        
        # Command history
        y = y_right + 50
        history_card = Card(50, y, self.screen_width - 100, 
                           self.screen_height - y - 20, "Command History")
        history_card.draw(self.screen)
        
        history_area = pygame.Rect(70, y + 50, history_card.rect.width - 40,
                                   history_card.rect.height - 70)
        pygame.draw.rect(self.screen, DesignSystem.COLORS['bg'], history_area,
                       border_radius=DesignSystem.RADIUS['sm'])
        
        font = DesignSystem.get_font('console')
        history_y = history_area.y + DesignSystem.SPACING['sm']
        for cmd in self.command_history[-15:]:
            cmd_surf = font.render(cmd, True, DesignSystem.COLORS['text_console'])
            if history_y + cmd_surf.get_height() > history_area.bottom - DesignSystem.SPACING['sm']:
                break
            self.screen.blit(cmd_surf, (history_area.x + DesignSystem.SPACING['md'], history_y))
            history_y += cmd_surf.get_height() + DesignSystem.SPACING['xs']
                
    def draw_pointcloud_tab(self):
        """Draw point cloud display tab using professional component with enhanced visuals."""
        # Get content area using layout manager
        content_area = self.layout.get_content_area()
        
        # Calculate header area with title and subtitle
        subtitle_text = "Use mouse to rotate, scroll to zoom, click cube to change view"
        header_rect, header_height = self.layout.calculate_header_area(
            content_area, 
            "Point Cloud Display",
            subtitle_text
        )
        
        # Draw title and subtitle
        title_label = Label(header_rect.x, header_rect.y, "Point Cloud Display", 
                           'title', DesignSystem.COLORS['text'])
        title_label.draw(self.screen)
        
        title_font = DesignSystem.get_font('title')
        subtitle_label = Label(header_rect.x, 
                              header_rect.y + title_font.get_height() + DesignSystem.SPACING['sm'],
                              subtitle_text, 'small', DesignSystem.COLORS['text_tertiary'])
        subtitle_label.draw(self.screen)
        
        # Calculate component area
        component_rect = self.layout.calculate_component_area(content_area, header_height, min_height=200)
        
        if not HAS_POINTCLOUD:
            # Enhanced error display using Card
            error_card_height = 200
            error_card = Card(component_rect.x, component_rect.y, component_rect.width, 
                            error_card_height, "Point Cloud Not Available")
            error_card.draw(self.screen)
            
            # Calculate error message positions using layout
            error_content_area = self.layout.calculate_inner_content_area(
                error_card.rect, has_title=True, padding='md'
            )
            error_center_x = error_content_area.centerx
            error_y = error_content_area.y + DesignSystem.SPACING['lg']
            
            error_label = Label(error_center_x, error_y,
                              "Point cloud display requires numpy and point cloud data",
                              'label', DesignSystem.COLORS['error'])
            error_label.align = 'center'
            error_label.draw(self.screen)
            
            install_label = Label(error_center_x, 
                                error_y + DesignSystem.get_font('label').get_height() + DesignSystem.SPACING['sm'],
                                "Please ensure numpy is installed: pip install numpy",
                                'small', DesignSystem.COLORS['text_tertiary'])
            install_label.align = 'center'
            install_label.draw(self.screen)
            return
        
        # Create Card to manage point cloud component and ensure it doesn't exceed bounds
        # Store card reference for event handling
        self.pc_card = Card(component_rect.x, component_rect.y, component_rect.width, 
                           component_rect.height, "Point Cloud 3D View")
        
        # Get Card's content area (below title) for point cloud component
        card_content_area = self.pc_card.get_content_area()
        
        # Set point cloud component position and size within Card's content area
        # Component coordinates are relative to Card's content area (0,0 at content area start)
        component_padding = DesignSystem.SPACING['sm']
        self.pointcloud_display.visible = True
        self.pointcloud_display.enabled = True
        self.pointcloud_display.rect.x = component_padding
        self.pointcloud_display.rect.y = component_padding
        self.pointcloud_display.rect.width = card_content_area.width - component_padding * 2
        self.pointcloud_display.rect.height = card_content_area.height - component_padding * 2
        # Don't show title in component since Card already has one
        self.pointcloud_display.title = ""
        
        # Update point cloud data and camera
        self.pointcloud_display.set_pointcloud(self.pc_surface_simple)
        self.pointcloud_display.set_camera(self.pc_camera_angle_x, 
                                          self.pc_camera_angle_y, 
                                          self.pc_zoom)
        
        # Calculate renderer size using layout manager (relative to component's inner area)
        if HAS_POINTCLOUD and self.pointcloud_display.rect.width > 0 and self.pointcloud_display.rect.height > 0:
            # Component doesn't have title, so calculate inner area without title offset
            component_inner_area = self.layout.calculate_inner_content_area(
                pygame.Rect(0, 0, self.pointcloud_display.rect.width, self.pointcloud_display.rect.height),
                has_title=False, 
                padding='sm'
            )
            renderer_width, renderer_height = self.layout.calculate_renderer_size(component_inner_area)
            
            # Create or update renderer if needed
            if self.pointcloud_display.renderer is None:
                self.pointcloud_display.renderer = PointCloudRenderer(
                    width=renderer_width, 
                    height=renderer_height
                )
            elif (self.pointcloud_display.renderer.width != renderer_width or 
                  self.pointcloud_display.renderer.height != renderer_height):
                self.pointcloud_display.renderer.width = renderer_width
                self.pointcloud_display.renderer.height = renderer_height
        
        # Add point cloud component to Card as child (Card will handle clipping)
        self.pc_card.add_child(self.pointcloud_display)
        
        # Draw Card (which will draw the point cloud component within bounds)
        self.pc_card.draw(self.screen)
        
        # Draw status indicator overlay (positioned relative to Card's content area)
        # Calculate indicator position within Card's content area
        indicator_height = 24
        indicator_width = 90
        indicator_y = component_padding + DesignSystem.SPACING['sm']
        
        # Check if indicator fits within Card's content area
        if indicator_y + indicator_height <= card_content_area.height:
            # Calculate absolute position for indicator (top-right of Card's content area)
            indicator_abs_x = self.pc_card.rect.right - DesignSystem.SPACING['md'] - indicator_width
            indicator_abs_y = self.pc_card.rect.y + LayoutManager.CARD_TITLE_HEIGHT + indicator_y
            
            # Ensure indicator is within Card bounds
            if (indicator_abs_x >= self.pc_card.rect.x and 
                indicator_abs_y >= self.pc_card.rect.y + LayoutManager.CARD_TITLE_HEIGHT and
                indicator_abs_x + indicator_width <= self.pc_card.rect.right and
                indicator_abs_y + indicator_height <= self.pc_card.rect.bottom):
                
                if self.pc_surface_simple is not None:
                    # Draw "Live" indicator
                    indicator_rect = pygame.Rect(indicator_abs_x, indicator_abs_y, indicator_width, indicator_height)
                    indicator_surf = pygame.Surface((indicator_rect.width, indicator_rect.height), pygame.SRCALPHA)
                    pygame.draw.rect(indicator_surf, (*DesignSystem.COLORS['success'], 180), 
                                   indicator_surf.get_rect(), border_radius=DesignSystem.RADIUS['sm'])
                    self.screen.blit(indicator_surf, indicator_rect)
                    
                    pygame.draw.rect(self.screen, DesignSystem.COLORS['success'], indicator_rect,
                                   width=1, border_radius=DesignSystem.RADIUS['sm'])
                    
                    font = DesignSystem.get_font('small')
                    live_text = "● LIVE"
                    live_surf = font.render(live_text, True, DesignSystem.COLORS['text'])
                    live_x = indicator_rect.centerx - live_surf.get_width() // 2
                    live_y = indicator_rect.centery - live_surf.get_height() // 2
                    self.screen.blit(live_surf, (live_x, live_y))
                else:
                    # Draw "Waiting" indicator
                    indicator_rect = pygame.Rect(indicator_abs_x, indicator_abs_y, indicator_width, indicator_height)
                    indicator_surf = pygame.Surface((indicator_rect.width, indicator_rect.height), pygame.SRCALPHA)
                    pygame.draw.rect(indicator_surf, (*DesignSystem.COLORS['text_tertiary'], 180), 
                                   indicator_surf.get_rect(), border_radius=DesignSystem.RADIUS['sm'])
                    self.screen.blit(indicator_surf, indicator_rect)
                    
                    pygame.draw.rect(self.screen, DesignSystem.COLORS['border'], indicator_rect,
                                   width=1, border_radius=DesignSystem.RADIUS['sm'])
                    
                    font = DesignSystem.get_font('small')
                    wait_text = "WAITING"
                    wait_surf = font.render(wait_text, True, DesignSystem.COLORS['text_tertiary'])
                    wait_x = indicator_rect.centerx - wait_surf.get_width() // 2
                    wait_y = indicator_rect.centery - wait_surf.get_height() // 2
                    self.screen.blit(wait_surf, (wait_x, wait_y))
    
    def draw_3d_view_tab(self):
        """Draw 3D view tab using Open3D offscreen rendering."""
        # Tab bar height is 45, add padding below it
        tab_height = 45
        y = tab_height + DesignSystem.SPACING['lg']
        
        title_label = Label(50, y, "3D Point Cloud View (Open3D)", 'title', DesignSystem.COLORS['text'])
        title_label.draw(self.screen)
        y += 50
        
        if not HAS_OPEN3D:
            error_card = Card(50, y, self.screen_width - 100, 200, "Open3D Not Available")
            error_card.draw(self.screen)
            
            error_label = Label(self.screen_width // 2 - 200, y + 100,
                              "Open3D is not installed. Please install it with: pip install open3d",
                              'label', DesignSystem.COLORS['error'])
            error_label.align = 'center'
            error_label.draw(self.screen)
            return
        
        # 3D view area
        view_card = Card(50, y, self.screen_width - 100, 
                        self.screen_height - y - 20, "3D Point Cloud")
        view_card.draw(self.screen)
        
        view_area = pygame.Rect(70, y + 50, view_card.rect.width - 40,
                               view_card.rect.height - 70)
        
        if self.pc_surface_o3d:
            # Scale and display Open3D rendered image
            img_rect = self.pc_surface_o3d.get_rect()
            scale = min(view_area.width / img_rect.width, view_area.height / img_rect.height)
            new_size = (int(img_rect.width * scale), int(img_rect.height * scale))
            scaled_img = pygame.transform.scale(self.pc_surface_o3d, new_size)
            scaled_rect = scaled_img.get_rect(center=view_area.center)
            self.screen.blit(scaled_img, scaled_rect)
        else:
            placeholder_label = Label(view_area.centerx - 150, view_area.centery - 10,
                                     "Waiting for point cloud data...", 'label',
                                     DesignSystem.COLORS['text_secondary'])
            placeholder_label.align = 'center'
            placeholder_label.draw(self.screen)
        
        # Instructions overlay
        instructions = [
            "Mouse Drag: Rotate view",
            "Scroll: Zoom in/out",
        ]
        
        font = DesignSystem.get_font('small')
        inst_y = y + 60
        for inst in instructions:
            inst_surf = font.render(inst, True, DesignSystem.COLORS['text'])
            self.screen.blit(inst_surf, (self.screen_width - 200, inst_y))
            inst_y += 20
                
    def draw_map_tab(self):
        """Draw map tab showing all drone positions using MapComponent."""
        # Tab bar height is 45, add padding below it
        tab_height = 45
        y = tab_height + DesignSystem.SPACING['lg']
        
        title_label = Label(50, y, "Drone Map View", 'title', DesignSystem.COLORS['text'])
        title_label.draw(self.screen)
        y += 50
        
        # Map display card
        self.map_card = Card(50, y, self.screen_width - 100, self.screen_height - y - 20, "Drone Positions")
        self.map_card.draw(self.screen)
        
        # Get card's content area for map component
        card_content_area = self.map_card.get_content_area()
        
        # Set map component position and size within Card's content area
        component_padding = DesignSystem.SPACING['sm']
        self.map_display.visible = True
        self.map_display.enabled = True
        self.map_display.rect.x = component_padding
        self.map_display.rect.y = component_padding
        self.map_display.rect.width = card_content_area.width - component_padding * 2
        self.map_display.rect.height = card_content_area.height - component_padding * 2
        # Don't show title in component since Card already has one
        self.map_display.title = ""
        
        # Update map component with current drones data
        self.map_display.set_drones(self.drones, self.current_drone_id)
        
        # Add map component to Card as child (Card will handle clipping and coordinate transformation)
        self.map_card.add_child(self.map_display)
        
        # Draw Card (which will draw the map component within bounds)
        self.map_card.draw(self.screen)
    
    def draw_network_tab(self):
        """Draw network test tab."""
        # Tab bar height is 45, add padding below it
        tab_height = 45
        y = tab_height + DesignSystem.SPACING['lg']
        
        title_label = Label(50, y, "Network Test", 'title', DesignSystem.COLORS['text'])
        title_label.draw(self.screen)
        y += 50
        
        config_card = Card(50, y, self.screen_width - 100, 200, "Test Configuration")
        config_card.draw(self.screen)
        
        card_y = y + 50
        url_label = Label(70, card_y, "Test Address:", 'label',
                         DesignSystem.COLORS['text_label'])
        url_label.draw(self.screen)
        self.test_url_input.rect.x = 70
        self.test_url_input.rect.y = card_y + 25
        self.test_url_input.rect.width = config_card.rect.width - 40
        self.test_url_input.draw(self.screen)
        card_y += 70
        
        timeout_label = Label(70, card_y, "Timeout (seconds):", 'label',
                            DesignSystem.COLORS['text_label'])
        timeout_label.draw(self.screen)
        self.test_timeout_input.rect.x = 70
        self.test_timeout_input.rect.y = card_y + 25
        self.test_timeout_input.rect.width = 200
        self.test_timeout_input.draw(self.screen)
        
        self.test_btn.rect.x = 280
        self.test_btn.rect.y = card_y + 25
        self.test_btn.draw(self.screen)
        
        y += config_card.rect.height + 20
        
        result_card = Card(50, y, self.screen_width - 100,
                          self.screen_height - y - 20, "Test Results")
        result_card.draw(self.screen)
        
        result_area = pygame.Rect(70, y + 50, result_card.rect.width - 40,
                                 result_card.rect.height - 70)
        pygame.draw.rect(self.screen, DesignSystem.COLORS['bg'], result_area,
                       border_radius=DesignSystem.RADIUS['sm'])
        
        font = DesignSystem.get_font('console')
        result_y = result_area.y + DesignSystem.SPACING['sm']
        for result in self.test_results[-25:]:
            result_surf = font.render(result, True, DesignSystem.COLORS['text_console'])
            if result_y + result_surf.get_height() > result_area.bottom - DesignSystem.SPACING['sm']:
                break
            self.screen.blit(result_surf, (result_area.x + DesignSystem.SPACING['md'], result_y))
            result_y += result_surf.get_height() + DesignSystem.SPACING['xs']
                
    def handle_events(self):
        """Handle all pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            # Tab switching
            if event.type == pygame.MOUSEBUTTONDOWN:
                tab_height = 45
                tab_width = self.screen_width // len(self.tabs)
                if event.pos[1] < tab_height:
                    tab_index = event.pos[0] // tab_width
                    if 0 <= tab_index < len(self.tabs):
                        self.current_tab = tab_index
                
            # Handle tab-specific events
            if self.current_tab == 0:  # Connection tab
                self.drone_name_input.handle_event(event)
                self.connection_url_input.handle_event(event)
                self.use_mock_checkbox.handle_event(event)
                self.add_drone_btn.handle_event(event)
                self.connect_btn.handle_event(event)
                self.disconnect_btn.handle_event(event)
                self.remove_drone_btn.handle_event(event)
                
                # Handle drone list selection
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    list_card_y = 45 + DesignSystem.SPACING['lg'] + 50 + 280 + 20
                    list_area = pygame.Rect(70, list_card_y + 50, self.screen_width - 140, 200)
                    if list_area.collidepoint(event.pos):
                        # Calculate which drone was clicked
                        relative_y = event.pos[1] - list_area.y
                        drone_index = relative_y // 35
                        if 0 <= drone_index < len(self.drones):
                            self.current_drone_id = list(self.drones.keys())[drone_index]
                            # Update input fields with selected drone info
                            drone_info = self.drones[self.current_drone_id]
                            self.drone_name_input.text = drone_info['name']
                            self.connection_url_input.text = drone_info['url']
                            self.use_mock_checkbox.checked = drone_info.get('use_mock', False)
            elif self.current_tab == 3:  # Control tab
                # Update button positions before handling events (same calculation as in draw_control_tab)
                tab_height = 45
                y = tab_height + DesignSystem.SPACING['lg']
                y += 50  # Title height
                
                x_right = 50 + 320 + 20  # topic_card width is 320
                y_right = y
                
                # Calculate positions same as in draw_control_tab
                y_right += 120 + 20  # config_card height
                y_right += 380 + 20  # editor_card height
                button_y = y_right
                button_x = x_right + 20
                
                for i, btn in enumerate(self.preset_buttons):
                    btn.rect.x = button_x + i * 140
                    btn.rect.y = button_y
                
                self.format_json_btn.rect.x = button_x + len(self.preset_buttons) * 140
                self.format_json_btn.rect.y = button_y
                
                self.send_command_btn.rect.x = button_x + len(self.preset_buttons) * 140 + 150
                self.send_command_btn.rect.y = button_y
                
                # Now handle events
                self.topic_list.handle_event(event)
                self.control_topic_input.handle_event(event)
                self.control_type_input.handle_event(event)
                self.json_editor.handle_event(event)
                for btn in self.preset_buttons:
                    btn.handle_event(event)
                self.format_json_btn.handle_event(event)
                self.send_command_btn.handle_event(event)
            elif self.current_tab == 1:  # Status tab - handle point cloud controls
                # Handle events through Card first (which will forward to children with correct coordinates)
                if HAS_POINTCLOUD and hasattr(self, 'status_pc_card') and self.status_pc_card:
                    if self.status_pc_card.handle_event(event):
                        # Card/component handled the event (e.g., cube click)
                        # Sync camera angles from component
                        angle_x, angle_y, zoom = self.status_pointcloud_display.get_camera()
                        self.pc_camera_angle_x = angle_x
                        self.pc_camera_angle_y = angle_y
                        self.pc_zoom = zoom
                        self.pc_render_cache_params = None
                        continue
                
                # Handle mouse interactions on status tab point cloud (fallback for non-component events)
                if HAS_POINTCLOUD and hasattr(event, 'pos') and hasattr(self, 'status_pc_card') and self.status_pc_card and self.status_pc_card.rect.collidepoint(event.pos):
                    current_time = time.time()
                    if current_time - self.pc_last_interaction_time < self.pc_interaction_throttle:
                        continue
                    
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 4:  # Scroll up - zoom in
                            self.pc_zoom = min(3.0, self.pc_zoom * 1.1)
                            self.status_pointcloud_display.set_camera(self.pc_camera_angle_x, 
                                                                     self.pc_camera_angle_y, 
                                                                     self.pc_zoom)
                            self.pc_last_interaction_time = current_time
                            self.pc_render_cache_params = None
                        elif event.button == 5:  # Scroll down - zoom out
                            self.pc_zoom = max(0.1, self.pc_zoom / 1.1)
                            self.status_pointcloud_display.set_camera(self.pc_camera_angle_x, 
                                                                     self.pc_camera_angle_y, 
                                                                     self.pc_zoom)
                            self.pc_last_interaction_time = current_time
                            self.pc_render_cache_params = None
                    elif event.type == pygame.MOUSEMOTION and hasattr(event, 'buttons') and event.buttons[0]:  # Drag to rotate
                        if hasattr(event, 'rel'):
                            self.pc_camera_angle_y += event.rel[0] * 0.01
                            self.pc_camera_angle_x += event.rel[1] * 0.01
                            self.pc_camera_angle_x = max(-math.pi/2, min(math.pi/2, self.pc_camera_angle_x))
                            self.status_pointcloud_display.set_camera(self.pc_camera_angle_x, 
                                                                     self.pc_camera_angle_y, 
                                                                     self.pc_zoom)
                            self.pc_last_interaction_time = current_time
                            # Invalidate cache to trigger re-render
                            self.pc_render_cache_params = None
            elif self.current_tab == 4:  # Point cloud tab - handle camera controls
                # Handle events through Card first (which will forward to children with correct coordinates)
                if self.pc_card and self.pc_card.handle_event(event):
                    # Card/component handled the event (e.g., cube click)
                    # Sync camera angles from component
                    angle_x, angle_y, zoom = self.pointcloud_display.get_camera()
                    self.pc_camera_angle_x = angle_x
                    self.pc_camera_angle_y = angle_y
                    self.pc_zoom = zoom
                    self.pc_render_cache_params = None
                    continue
                
                current_time = time.time()
                # Throttle interactions for performance
                if current_time - self.pc_last_interaction_time < self.pc_interaction_throttle:
                    continue
                
                # Handle mouse interactions on point cloud area
                # Convert event position to component's coordinate system (relative to Card's content area)
                if hasattr(event, 'pos') and self.pc_card and self.pc_card.rect.collidepoint(event.pos):
                    # Convert to Card's content area coordinates
                    card_content_area = self.pc_card.get_content_area()
                    rel_x = event.pos[0] - card_content_area.x
                    rel_y = event.pos[1] - card_content_area.y
                    
                    # Check if within component bounds
                    if self.pointcloud_display.rect.collidepoint((rel_x, rel_y)):
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            if event.button == 4:  # Scroll up - zoom in
                                self.pc_zoom = min(3.0, self.pc_zoom * 1.1)
                                self.pointcloud_display.set_camera(self.pc_camera_angle_x, 
                                                                  self.pc_camera_angle_y, 
                                                                  self.pc_zoom)
                                self.pc_last_interaction_time = current_time
                                self.pc_render_cache_params = None
                            elif event.button == 5:  # Scroll down - zoom out
                                self.pc_zoom = max(0.1, self.pc_zoom / 1.1)
                                self.pointcloud_display.set_camera(self.pc_camera_angle_x, 
                                                                  self.pc_camera_angle_y, 
                                                                  self.pc_zoom)
                                self.pc_last_interaction_time = current_time
                                self.pc_render_cache_params = None
                        elif event.type == pygame.MOUSEMOTION and event.buttons[0]:  # Drag to rotate
                            self.pc_camera_angle_y += event.rel[0] * 0.01
                            self.pc_camera_angle_x += event.rel[1] * 0.01
                            self.pc_camera_angle_x = max(-math.pi/2, min(math.pi/2, self.pc_camera_angle_x))
                            self.pointcloud_display.set_camera(self.pc_camera_angle_x, 
                                                              self.pc_camera_angle_y, 
                                                              self.pc_zoom)
                            self.pc_last_interaction_time = current_time
                            # Invalidate cache to trigger re-render
                            self.pc_render_cache_params = None
            elif self.current_tab == 1:  # Status tab - handle point cloud controls
                # Handle status tab point cloud component
                if HAS_POINTCLOUD and self.status_pointcloud_display.handle_event(event):
                    # Component handled the event (e.g., cube click)
                    # Sync camera angles from component
                    angle_x, angle_y, zoom = self.status_pointcloud_display.get_camera()
                    self.pc_camera_angle_x = angle_x
                    self.pc_camera_angle_y = angle_y
                    self.pc_zoom = zoom
                    self.pc_render_cache_params = None
                    continue
                
                # Handle mouse interactions on status tab point cloud
                if HAS_POINTCLOUD and hasattr(event, 'pos') and self.status_pointcloud_display.rect.collidepoint(event.pos):
                    current_time = time.time()
                    if current_time - self.pc_last_interaction_time < self.pc_interaction_throttle:
                        continue
                    
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 4:  # Scroll up - zoom in
                            self.pc_zoom = min(3.0, self.pc_zoom * 1.1)
                            self.status_pointcloud_display.set_camera(self.pc_camera_angle_x, 
                                                                     self.pc_camera_angle_y, 
                                                                     self.pc_zoom)
                            self.pc_last_interaction_time = current_time
                            self.pc_render_cache_params = None
                        elif event.button == 5:  # Scroll down - zoom out
                            self.pc_zoom = max(0.1, self.pc_zoom / 1.1)
                            self.status_pointcloud_display.set_camera(self.pc_camera_angle_x, 
                                                                     self.pc_camera_angle_y, 
                                                                     self.pc_zoom)
                            self.pc_last_interaction_time = current_time
                            self.pc_render_cache_params = None
                    elif event.type == pygame.MOUSEMOTION and hasattr(event, 'buttons') and event.buttons[0]:  # Drag to rotate
                        if hasattr(event, 'rel'):
                            self.pc_camera_angle_y += event.rel[0] * 0.01
                            self.pc_camera_angle_x += event.rel[1] * 0.01
                            self.pc_camera_angle_x = max(-math.pi/2, min(math.pi/2, self.pc_camera_angle_x))
                            self.status_pointcloud_display.set_camera(self.pc_camera_angle_x, 
                                                                     self.pc_camera_angle_y, 
                                                                     self.pc_zoom)
                            self.pc_last_interaction_time = current_time
                            # Invalidate cache to trigger re-render
                            self.pc_render_cache_params = None
            elif self.current_tab == 5:  # 3D View tab - handle Open3D controls
                if HAS_OPEN3D and hasattr(event, 'pos'):
                    view_area = pygame.Rect(70, 110, self.screen_width - 140, self.screen_height - 130)
                    if view_area.collidepoint(event.pos) and self.o3d_vis:
                        current_time = time.time()
                        # Throttle interactions for performance
                        if current_time - self.pc_last_interaction_time < self.pc_interaction_throttle:
                            return
                        
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            if event.button == 4:  # Scroll up - zoom in
                                ctr = self.o3d_vis.get_view_control()
                                ctr.scale(0.9)
                                self.pc_last_interaction_time = current_time
                            elif event.button == 5:  # Scroll down - zoom out
                                ctr = self.o3d_vis.get_view_control()
                                ctr.scale(1.1)
                                self.pc_last_interaction_time = current_time
                        elif event.type == pygame.MOUSEMOTION and hasattr(event, 'buttons') and event.buttons[0]:  # Drag to rotate
                            if hasattr(event, 'rel'):
                                ctr = self.o3d_vis.get_view_control()
                                ctr.rotate(event.rel[0] * 0.5, event.rel[1] * 0.5)
                                self.pc_last_interaction_time = current_time
            elif self.current_tab == 6:  # Map tab
                # Handle events through Card first (which will forward to children with correct coordinates)
                if hasattr(self, 'map_card') and self.map_card and self.map_card.handle_event(event):
                    # Card/component handled the event
                    continue
                
                # Map interactions can be added here (zoom, pan, click to select drone, etc.)
                if hasattr(event, 'pos') and hasattr(self, 'map_card') and self.map_card and self.map_card.rect.collidepoint(event.pos):
                    # Could implement click-to-select drone here
                    pass
            elif self.current_tab == 7:  # Network tab
                self.test_url_input.handle_event(event)
                self.test_timeout_input.handle_event(event)
                self.test_btn.handle_event(event)
                
    def update(self):
        """Update game state."""
        self.dt = self.clock.tick(60) / 1000.0
        
        # Update UI components
        if self.current_tab == 0:
            self.connection_url_input.update(self.dt)
            self.connect_btn.update(self.dt)
            self.disconnect_btn.update(self.dt)
        elif self.current_tab == 1:
            # Update status tab components
            if HAS_POINTCLOUD:
                self.status_pointcloud_display.update(self.dt)
        elif self.current_tab == 2:
            # Update image tab component
            self.image_display.update(self.dt)
        elif self.current_tab == 3:
            self.control_topic_input.update(self.dt)
            self.control_type_input.update(self.dt)
            self.json_editor.update(self.dt)
            for btn in self.preset_buttons:
                btn.update(self.dt)
            self.format_json_btn.update(self.dt)
            self.send_command_btn.update(self.dt)
        elif self.current_tab == 4:
            # Update point cloud tab component
            self.pointcloud_display.update(self.dt)
        elif self.current_tab == 6:
            # Map tab - update map component
            if self.map_display:
                self.map_display.update(self.dt)
        elif self.current_tab == 7:
            self.test_url_input.update(self.dt)
            self.test_timeout_input.update(self.dt)
            self.test_btn.update(self.dt)
            
    def draw(self):
        """Draw everything."""
        # Clear screen
        self.screen.fill(DesignSystem.COLORS['bg'])
        
        # Draw subtle grid pattern
        for y in range(0, self.screen_height, 60):
            pygame.draw.line(self.screen, DesignSystem.COLORS['bg_secondary'],
                           (0, y), (self.screen_width, y), 1)
        
        # Draw tab bar
        self.draw_tab_bar()
        
        # Draw tab content
        if self.current_tab == 0:
            self.draw_connection_tab()
        elif self.current_tab == 1:
            self.draw_status_tab()
        elif self.current_tab == 2:
            self.draw_image_tab()
        elif self.current_tab == 3:
            self.draw_control_tab()
        elif self.current_tab == 4:  # Point cloud tab
            self.draw_pointcloud_tab()
        elif self.current_tab == 5:
            self.draw_3d_view_tab()
        elif self.current_tab == 6:
            self.draw_map_tab()
        elif self.current_tab == 7:
            self.draw_network_tab()
            
        pygame.display.flip()
        
    def run(self):
        """Main game loop."""
        if not hasattr(self, 'connection_logs'):
            self.connection_logs = []
        self.add_log("Waiting for connection...")
        
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            
        # Cleanup
        self.stop_update.set()
        self.stop_render_threads.set()
        
        # Disconnect all drones
        for drone_id, drone_info in self.drones.items():
            if drone_info.get('client'):
                try:
                    drone_info['client'].terminate()
                except:
                    pass
        
        # Legacy cleanup
        if self.client:
            try:
                self.client.terminate()
            except:
                pass
        
        # Close Open3D window if created
        if HAS_OPEN3D and self.o3d_window_created and self.o3d_vis:
            try:
                self.o3d_vis.destroy_window()
            except:
                pass
        pygame.quit()


def main():
    """Main entry point."""
    app = RosClientPygameGUI()
    app.run()


if __name__ == "__main__":
    main()
