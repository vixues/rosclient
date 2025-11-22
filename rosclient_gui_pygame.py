#!/usr/bin/env python
"""Industrial-grade GUI for RosClient using Pygame with fighter cockpit design."""
import pygame
import threading
import json
import time
from typing import Optional, Dict, Any, List, Tuple, Callable
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

class UIComponent:
    """Base class for all UI components."""
    
    def __init__(self, x: int, y: int, width: int, height: int):
        self.rect = pygame.Rect(x, y, width, height)
        self.visible = True
        self.enabled = True
        
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
            if self.rect.collidepoint(event.pos):
                rel_event = pygame.event.Event(event.type)
                rel_event.pos = (event.pos[0] - self.rect.x, event.pos[1] - self.rect.y)
                rel_event.button = getattr(event, 'button', None)
                
                for child in reversed(self.children):
                    if child.handle_event(rel_event):
                        return True
        return False
        
    def update(self, dt: float):
        """Update card and children."""
        for child in self.children:
            child.update(dt)
            
    def draw(self, surface: pygame.Surface):
        """Draw card with fighter cockpit styling."""
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
        
        # Draw title header if present
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
        
        # Draw children
        for child in self.children:
            child_surface = surface.subsurface(
                pygame.Rect(child.rect.x + self.rect.x,
                           child.rect.y + self.rect.y,
                           child.rect.width, child.rect.height)
            )
            child.draw(child_surface)


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
        self.callback = callback
        self.color = color or DesignSystem.COLORS['primary']
        self.hovered = False
        self.pressed = False
        self.animation_scale = 1.0
        
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle button events."""
        if not self.visible or not self.enabled:
            return False
            
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.pressed = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            if self.pressed and self.rect.collidepoint(event.pos):
                if self.callback:
                    self.callback()
                self.pressed = False
                return True
            self.pressed = False
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
        """Handle text input events."""
        if not self.visible or not self.enabled:
            return False
            
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
            return self.active
        elif event.type == pygame.KEYDOWN and self.active:
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
        self.callback = callback
        self.hovered = False
        
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle checkbox events."""
        if not self.visible or not self.enabled:
            return False
            
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.checked = not self.checked
                if self.callback:
                    self.callback(self.checked)
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
# ADVANCED COMPONENTS
# ============================================================================

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
        lines = self.text.split('\n') if self.text else [""]
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
        
        # Client state
        self.client: Optional[RosClient] = None
        self.is_connected = False
        self.update_thread: Optional[threading.Thread] = None
        self.stop_update = threading.Event()
        self.image_queue = queue.Queue(maxsize=1)
        self.current_image = None
        
        # Tabs
        self.tabs = ["Connection", "Status", "Image", "Control", "Point Cloud", "3D View", "Network Test"]
        self.current_tab = 0
        
        # UI Components
        self.setup_ui()
        self.setup_update_loop()
        
    def setup_ui(self):
        """Setup UI components using new component system."""
        # Connection tab components
        self.connection_url_input = TextInput(200, 100, 400, 35, "ws://localhost:9090")
        self.use_mock_checkbox = Checkbox(200, 150, "Use Mock Client (Test Mode)", False)
        self.connect_btn = Button(200, 200, 120, 40, "Connect", self.connect)
        self.disconnect_btn = Button(330, 200, 120, 40, "Disconnect", self.disconnect)
        self.disconnect_btn.color = DesignSystem.COLORS['error']
        
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
        
        # Point cloud components
        self.current_point_cloud = None
        self.pc_surface = None
        self.pc_surface_simple = None  # Simple rendering for status/pointcloud tabs
        self.pc_surface_o3d = None  # Open3D rendering surface
        self.pc_camera_angle_x = 0.0
        self.pc_camera_angle_y = 0.0
        self.pc_zoom = 1.0
        self.pc_last_render_time = 0.0
        self.pc_render_interval = 0.05  # ~20 FPS for point cloud updates (reduced for performance)
        self.pc_last_interaction_time = 0.0
        self.pc_interaction_throttle = 0.016  # ~60 FPS for mouse interactions
        
        # Open3D components (offscreen rendering)
        self.o3d_vis = None
        self.o3d_geometry = None
        self.o3d_window_created = False
        self.o3d_last_update = 0.0
        self.o3d_update_interval = 0.05  # ~20 FPS for Open3D updates
        
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
        
    def on_topic_selected(self, topic_data: Tuple[str, str]):
        """Handle topic selection."""
        if isinstance(topic_data, tuple):
            self.control_topic_input.text = topic_data[0]
            if topic_data[1]:
                self.control_type_input.text = topic_data[1]
        else:
            self.control_topic_input.text = topic_data
        
    def setup_update_loop(self):
        """Setup periodic update loop."""
        def update_loop():
            while not self.stop_update.is_set():
                try:
                    if self.is_connected and self.client:
                        if self.current_tab == 1:  # Status tab
                            self.update_status()
                            if HAS_CV2:  # Update image in status tab
                                self.update_image()
                            if HAS_POINTCLOUD:  # Update point cloud in status tab (simple rendering)
                                self.update_pointcloud_simple()
                                # Auto-rotate camera (smooth rotation)
                                self.pc_camera_angle_y += 0.005
                        if self.current_tab == 2 and HAS_CV2:  # Image tab
                            self.update_image()
                        if self.current_tab == 4 and HAS_POINTCLOUD:  # Point cloud tab (simple rendering)
                            self.update_pointcloud_simple()
                            # Auto-rotate camera (smooth rotation)
                            self.pc_camera_angle_y += 0.005
                        if self.current_tab == 5 and HAS_OPEN3D:  # 3D View tab (Open3D)
                            self.update_pointcloud_open3d()
                        if self.current_tab == 3:  # Control tab
                            self.update_topic_list()
                except Exception as e:
                    print(f"Update error: {e}")
                time.sleep(1)
                
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
        
    def connect(self):
        """Connect to ROS bridge."""
        url = self.connection_url_input.text.strip()
        if not url:
            self.add_log("Error: Please enter WebSocket address")
            return
            
        def connect_thread():
            try:
                self.add_log(f"Connecting to {url}...")
                
                if self.use_mock_checkbox.checked:
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
        if not hasattr(self, 'connection_logs'):
            self.connection_logs = []
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
            
    def update_image(self):
        """Update image display."""
        if not self.client or not self.is_connected:
            return
            
        try:
            image_data = self.client.get_latest_image()
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
                self.current_image = pygame.surfarray.make_surface(frame_flipped)
        except Exception:
            pass
            
    def update_pointcloud_simple(self):
        """Update point cloud display with simple rendering (for status/pointcloud tabs)."""
        if not self.client or not self.is_connected:
            return
            
        try:
            current_time = time.time()
            # Throttle rendering to maintain smooth performance
            if current_time - self.pc_last_render_time < self.pc_render_interval:
                return
                
            pc_data = self.client.get_latest_point_cloud()
            if pc_data:
                points, timestamp = pc_data
                self.current_point_cloud = points
                self.render_pointcloud_simple()
                self.pc_last_render_time = current_time
        except Exception:
            pass
    
    def update_pointcloud_open3d(self):
        """Update point cloud display using Open3D (for 3D View tab)."""
        if not self.client or not self.is_connected:
            return
            
        try:
            pc_data = self.client.get_latest_point_cloud()
            if pc_data:
                points, timestamp = pc_data
                self.current_point_cloud = points
                self.render_pointcloud_open3d()
        except Exception:
            pass
            
    def render_pointcloud_simple(self):
        """Render point cloud using simple 3D projection (original method for status/pointcloud tabs)."""
        if not HAS_POINTCLOUD or self.current_point_cloud is None:
            return
            
        try:
            points = self.current_point_cloud
            if len(points) == 0:
                return
                
            # Ensure numpy is available
            try:
                import numpy as np
            except ImportError:
                print("Warning: numpy required for point cloud rendering")
                return
            
            # Convert to numpy array if not already
            if not isinstance(points, np.ndarray):
                points = np.array(points, dtype=np.float32)
                
            # Sample points if too many for performance (reduced for better performance)
            max_points = 10000  # Reduced from 20000 for better performance
            if len(points) > max_points:
                step = len(points) // max_points
                points = points[::step]
                
            # Create surface for rendering
            width, height = 800, 600
            self.pc_surface_simple = pygame.Surface((width, height))
            self.pc_surface_simple.fill(DesignSystem.COLORS['bg'])
            
            # Calculate center and scale
            center = np.mean(points, axis=0)
            points_centered = points - center
            distances = np.linalg.norm(points_centered, axis=1)
            max_dist = np.max(distances) if len(distances) > 0 else 1.0
            if max_dist == 0:
                max_dist = 1.0
                
            scale = min(width, height) * 0.4 / max_dist * self.pc_zoom
            
            # Rotate points based on camera angles
            cos_x, sin_x = math.cos(self.pc_camera_angle_x), math.sin(self.pc_camera_angle_x)
            cos_y, sin_y = math.cos(self.pc_camera_angle_y), math.sin(self.pc_camera_angle_y)
            
            # Project and draw points (simple method)
            x, y, z = points_centered[:, 0], points_centered[:, 1], points_centered[:, 2]
            
            # Rotate around X axis
            y_rot = y * cos_x - z * sin_x
            z_rot = y * sin_x + z * cos_x
            
            # Rotate around Y axis
            x_final = x * cos_y + z_rot * sin_y
            z_final = -x * sin_y + z_rot * cos_y
            
            # Filter points in front
            front_mask = z_final > -max_dist * 0.1
            x_final = x_final[front_mask]
            y_final = y_rot[front_mask]
            z_final = z_final[front_mask]
            
            if len(x_final) == 0:
                return
            
            # Simple perspective projection
            z_scale = 1.0 + z_final / max_dist
            proj_x = (width / 2 + x_final * scale / z_scale).astype(np.int32)
            proj_y = (height / 2 - y_final * scale / z_scale).astype(np.int32)
            
            # Filter points within bounds
            valid_mask = (proj_x >= 0) & (proj_x < width) & (proj_y >= 0) & (proj_y < height)
            proj_x = proj_x[valid_mask]
            proj_y = proj_y[valid_mask]
            z_final = z_final[valid_mask]
            
            if len(proj_x) == 0:
                return
            
            # Color based on depth
            z_normalized = np.clip((z_final + max_dist) / (2 * max_dist), 0.0, 1.0)
            primary_r, primary_g, primary_b = DesignSystem.COLORS['primary']
            colors_r = (primary_r * z_normalized).astype(np.uint8)
            colors_g = (primary_g * z_normalized).astype(np.uint8)
            colors_b = np.full(len(z_normalized), primary_b, dtype=np.uint8)
            
            # Draw points using pixel array for better performance
            pixel_array = pygame.surfarray.pixels3d(self.pc_surface_simple)
            for i in range(len(proj_x)):
                px, py = int(proj_x[i]), int(proj_y[i])
                if 0 <= px < width and 0 <= py < height:
                    color = (int(colors_r[i]), int(colors_g[i]), int(colors_b[i]))
                    pixel_array[py, px] = color
            del pixel_array  # Unlock surface
            
            # Draw axes
            axis_length = max_dist * 0.3
            origin_x, origin_y = width // 2, height // 2
            
            axis_points = np.array([
                [axis_length, 0, 0],
                [0, axis_length, 0],
                [0, 0, axis_length],
            ])
            
            axis_colors = [
                DesignSystem.COLORS['error'],
                DesignSystem.COLORS['success'],
                DesignSystem.COLORS['primary'],
            ]
            
            for axis_point, color in zip(axis_points, axis_colors):
                axis_2d = self.project_point_3d(axis_point, center, max_dist, scale, width, height)
                if axis_2d:
                    pygame.draw.line(self.pc_surface_simple, color, 
                                   (origin_x, origin_y), axis_2d, 2)
            
            # Draw info text
            font = DesignSystem.get_font('small')
            info_text = f"Points: {len(points)} | Zoom: {self.pc_zoom:.2f}"
            text_surf = font.render(info_text, True, DesignSystem.COLORS['text_secondary'])
            self.pc_surface_simple.blit(text_surf, (10, 10))
            
        except Exception as e:
            print(f"Point cloud simple render error: {e}")
    
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
            
            points = self.current_point_cloud
            if len(points) == 0:
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
            
            # Create or update offscreen visualizer
            if not self.o3d_window_created:
                self.o3d_vis = o3d.visualization.Visualizer()
                self.o3d_vis.create_window("3D Point Cloud View", width=1200, height=800, visible=False)
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
            
            # Draw info text
            font = DesignSystem.get_font('small')
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
            
    def set_preset_command(self, command: str):
        """Set preset command."""
        self.json_editor.text = command
        self.json_editor.cursor_pos = [0, 0]
        
    def format_json(self):
        """Format JSON in editor."""
        self.json_editor.format_json()
        
    def update_topic_list(self):
        """Update topic list from ROS."""
        try:
            from rosclient.clients.config import DEFAULT_TOPICS
            topics = [(topic.name, topic.type) for topic in DEFAULT_TOPICS.values()]
            
            if self.client and self.is_connected:
                if hasattr(self.client, '_ts_mgr') and self.client._ts_mgr:
                    if hasattr(self.client._ts_mgr, '_topics'):
                        for topic_name in self.client._ts_mgr._topics.keys():
                            if not any(t[0] == topic_name for t in topics):
                                topics.append((topic_name, ""))
            
            topics = sorted(topics, key=lambda x: x[0])
            self.topic_list.set_items(topics)
        except Exception as e:
            self.topic_list.set_items([])
        
    def send_control_command(self):
        """Send control command."""
        if not self.client or not self.is_connected:
            self.add_log("Warning: Please connect first")
            return
            
        try:
            topic = self.control_topic_input.text.strip()
            topic_type = self.control_type_input.text.strip()
            message_text = self.json_editor.text.strip()
            
            if not topic or not message_text:
                self.add_log("Warning: Please fill in Topic and message content")
                return
                
            try:
                message = json.loads(message_text)
            except json.JSONDecodeError as e:
                self.add_log(f"Error: JSON format error: {e}")
                return
                
            self.client.publish(topic, topic_type, message)
            
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
        tab_height = 45
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
        """Draw connection configuration tab."""
        y = 60
        
        # Title
        title_label = Label(50, y, "Connection Configuration", 'title', 
                           DesignSystem.COLORS['text'])
        title_label.draw(self.screen)
        y += 50
        
        # Connection settings card
        settings_card = Card(50, y, self.screen_width - 100, 220, "Connection Settings")
        settings_card.draw(self.screen)
        
        card_y = y + 50
        # Connection URL
        url_label = Label(70, card_y, "WebSocket Address:", 'label',
                         DesignSystem.COLORS['text_label'])
        url_label.draw(self.screen)
        self.connection_url_input.rect.x = 70
        self.connection_url_input.rect.y = card_y + 25
        self.connection_url_input.rect.width = settings_card.rect.width - 40
        self.connection_url_input.draw(self.screen)
        card_y += 70
        
        # Mock checkbox
        self.use_mock_checkbox.rect.x = 70
        self.use_mock_checkbox.rect.y = card_y
        self.use_mock_checkbox.draw(self.screen)
        card_y += 40
        
        # Buttons
        self.connect_btn.rect.x = 70
        self.connect_btn.rect.y = card_y
        self.connect_btn.draw(self.screen)
        self.disconnect_btn.rect.x = 200
        self.disconnect_btn.rect.y = card_y
        self.disconnect_btn.draw(self.screen)
        
        y += settings_card.rect.height + 20
        
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
        y = 60
        
        # Title with connection indicator
        title_label = Label(50, y, "Status Monitoring", 'title',
                           DesignSystem.COLORS['text'])
        title_label.draw(self.screen)
        
        # Connection indicator
        indicator_color = (DesignSystem.COLORS['success'] if self.is_connected 
                          else DesignSystem.COLORS['error'])
        indicator_text = "Connected" if self.is_connected else "Disconnected"
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
        y = status_bottom + DesignSystem.SPACING['lg']
        
        # Camera and Point Cloud display (side by side)
        display_height = self.screen_height - y - 20
        display_width = (self.screen_width - 100 - DesignSystem.SPACING['md']) // 2
        
        # Camera display (left)
        if HAS_CV2:
            camera_card = Card(50, y, display_width, display_height, "Camera Stream")
            camera_card.draw(self.screen)
            
            img_area = pygame.Rect(70, y + 50, display_width - 40, display_height - 70)
            
            if self.current_image:
                img_rect = self.current_image.get_rect()
                img_rect.center = img_area.center
                if img_rect.width > img_area.width or img_rect.height > img_area.height:
                    scale = min(img_area.width / img_rect.width, img_area.height / img_rect.height)
                    new_size = (int(img_rect.width * scale), int(img_rect.height * scale))
                    img_scaled = pygame.transform.scale(self.current_image, new_size)
                    img_rect = img_scaled.get_rect(center=img_area.center)
                    self.screen.blit(img_scaled, img_rect)
                else:
                    self.screen.blit(self.current_image, img_rect)
            else:
                placeholder_label = Label(img_area.centerx - 100, img_area.centery - 10,
                                         "Waiting for image...", 'label',
                                         DesignSystem.COLORS['text_secondary'])
                placeholder_label.align = 'center'
                placeholder_label.draw(self.screen)
        
        # Point Cloud display (right)
        if HAS_POINTCLOUD:
            pc_x = 50 + display_width + DesignSystem.SPACING['md']
            pc_card = Card(pc_x, y, display_width, display_height, "Point Cloud")
            pc_card.draw(self.screen)
            
            pc_area = pygame.Rect(pc_x + 20, y + 50, display_width - 40, display_height - 70)
            
            if self.pc_surface_simple:
                # Create a copy to avoid surface locking issues during blit
                pc_copy = self.pc_surface_simple.copy()
                pc_rect = pc_copy.get_rect()
                scale = min(pc_area.width / pc_rect.width, pc_area.height / pc_rect.height)
                new_size = (int(pc_rect.width * scale), int(pc_rect.height * scale))
                scaled_pc = pygame.transform.scale(pc_copy, new_size)
                scaled_rect = scaled_pc.get_rect(center=pc_area.center)
                self.screen.blit(scaled_pc, scaled_rect)
            else:
                placeholder_label = Label(pc_area.centerx - 100, pc_area.centery - 10,
                                         "Waiting for point cloud...", 'label',
                                         DesignSystem.COLORS['text_secondary'])
                placeholder_label.align = 'center'
                placeholder_label.draw(self.screen)
            
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
        """Draw image display tab."""
        y = 60
        
        title_label = Label(50, y, "Image Display", 'title', DesignSystem.COLORS['text'])
        title_label.draw(self.screen)
        y += 50
        
        img_card = Card(50, y, self.screen_width - 100, 
                       self.screen_height - y - 20, "Image Stream")
        img_card.draw(self.screen)
        
        img_area = pygame.Rect(70, y + 50, img_card.rect.width - 40,
                              img_card.rect.height - 70)
        
        if self.current_image:
            img_rect = self.current_image.get_rect()
            img_rect.center = img_area.center
            if img_rect.width > img_area.width or img_rect.height > img_area.height:
                scale = min(img_area.width / img_rect.width, img_area.height / img_rect.height)
                new_size = (int(img_rect.width * scale), int(img_rect.height * scale))
                self.current_image = pygame.transform.scale(self.current_image, new_size)
                img_rect = self.current_image.get_rect(center=img_area.center)
            self.screen.blit(self.current_image, img_rect)
        else:
            placeholder_label = Label(img_area.centerx - 150, img_area.centery - 10,
                                     "Waiting for image data...", 'label',
                                     DesignSystem.COLORS['text_secondary'])
            placeholder_label.align = 'center'
            placeholder_label.draw(self.screen)
            
    def draw_control_tab(self):
        """Draw control command tab."""
        y = 60
        
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
        """Draw point cloud display tab."""
        y = 60
        
        title_label = Label(50, y, "Point Cloud Display", 'title', DesignSystem.COLORS['text'])
        title_label.draw(self.screen)
        y += 50
        
        if not HAS_POINTCLOUD:
            error_label = Label(self.screen_width // 2 - 200, y + 50,
                              "Point cloud display not available",
                              'label', DesignSystem.COLORS['error'])
            error_label.align = 'center'
            error_label.draw(self.screen)
            return
            
        pc_card = Card(50, y, self.screen_width - 100, 
                      self.screen_height - y - 20, "Point Cloud Stream")
        pc_card.draw(self.screen)
        
        pc_area = pygame.Rect(70, y + 50, pc_card.rect.width - 40,
                             pc_card.rect.height - 70)
        
        if self.pc_surface_simple:
            # Create a copy to avoid surface locking issues during blit
            pc_copy = self.pc_surface_simple.copy()
            pc_rect = pc_copy.get_rect()
            scale = min(pc_area.width / pc_rect.width, pc_area.height / pc_rect.height)
            new_size = (int(pc_rect.width * scale), int(pc_rect.height * scale))
            scaled_pc = pygame.transform.scale(pc_copy, new_size)
            scaled_rect = scaled_pc.get_rect(center=pc_area.center)
            self.screen.blit(scaled_pc, scaled_rect)
        else:
            placeholder_label = Label(pc_area.centerx - 150, pc_area.centery - 10,
                                     "Waiting for point cloud data...", 'label',
                                     DesignSystem.COLORS['text_secondary'])
            placeholder_label.align = 'center'
            placeholder_label.draw(self.screen)
    
    def draw_3d_view_tab(self):
        """Draw 3D view tab using Open3D offscreen rendering."""
        y = 60
        
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
                
    def draw_network_tab(self):
        """Draw network test tab."""
        y = 60
        
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
                self.connection_url_input.handle_event(event)
                self.use_mock_checkbox.handle_event(event)
                self.connect_btn.handle_event(event)
                self.disconnect_btn.handle_event(event)
            elif self.current_tab == 3:  # Control tab
                self.topic_list.handle_event(event)
                self.control_topic_input.handle_event(event)
                self.control_type_input.handle_event(event)
                self.json_editor.handle_event(event)
                for btn in self.preset_buttons:
                    btn.handle_event(event)
                self.format_json_btn.handle_event(event)
                self.send_command_btn.handle_event(event)
            elif self.current_tab == 1:  # Status tab - handle point cloud controls
                # Check if mouse is over point cloud area (only for mouse events)
                if HAS_POINTCLOUD and hasattr(event, 'pos'):
                    display_height = self.screen_height - 400
                    display_width = (self.screen_width - 100 - DesignSystem.SPACING['md']) // 2
                    pc_x = 50 + display_width + DesignSystem.SPACING['md']
                    pc_y = 400
                    pc_area = pygame.Rect(pc_x + 20, pc_y + 50, display_width - 40, display_height - 70)
                    
                    if pc_area.collidepoint(event.pos):
                        current_time = time.time()
                        # Throttle interactions for performance
                        if current_time - self.pc_last_interaction_time < self.pc_interaction_throttle:
                            return
                        
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            if event.button == 4:  # Scroll up - zoom in
                                self.pc_zoom = min(3.0, self.pc_zoom * 1.1)
                                self.pc_last_interaction_time = current_time
                            elif event.button == 5:  # Scroll down - zoom out
                                self.pc_zoom = max(0.1, self.pc_zoom / 1.1)
                                self.pc_last_interaction_time = current_time
                        elif event.type == pygame.MOUSEMOTION and hasattr(event, 'buttons') and event.buttons[0]:  # Drag to rotate
                            if hasattr(event, 'rel'):
                                self.pc_camera_angle_y += event.rel[0] * 0.01
                                self.pc_camera_angle_x += event.rel[1] * 0.01
                                self.pc_camera_angle_x = max(-math.pi/2, min(math.pi/2, self.pc_camera_angle_x))
                                self.pc_last_interaction_time = current_time
            elif self.current_tab == 4:  # Point cloud tab - handle camera controls
                current_time = time.time()
                # Throttle interactions for performance
                if current_time - self.pc_last_interaction_time < self.pc_interaction_throttle:
                    return
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:  # Scroll up - zoom in
                        self.pc_zoom = min(3.0, self.pc_zoom * 1.1)
                        self.pc_last_interaction_time = current_time
                    elif event.button == 5:  # Scroll down - zoom out
                        self.pc_zoom = max(0.1, self.pc_zoom / 1.1)
                        self.pc_last_interaction_time = current_time
                elif event.type == pygame.MOUSEMOTION and event.buttons[0]:  # Drag to rotate
                    self.pc_camera_angle_y += event.rel[0] * 0.01
                    self.pc_camera_angle_x += event.rel[1] * 0.01
                    self.pc_camera_angle_x = max(-math.pi/2, min(math.pi/2, self.pc_camera_angle_x))
                    self.pc_last_interaction_time = current_time
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
            elif self.current_tab == 6:  # Network tab
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
        elif self.current_tab == 3:
            self.control_topic_input.update(self.dt)
            self.control_type_input.update(self.dt)
            self.json_editor.update(self.dt)
            for btn in self.preset_buttons:
                btn.update(self.dt)
            self.format_json_btn.update(self.dt)
            self.send_command_btn.update(self.dt)
        elif self.current_tab == 6:
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
        elif self.current_tab == 4:
            self.draw_pointcloud_tab()
        elif self.current_tab == 5:
            self.draw_3d_view_tab()
        elif self.current_tab == 6:
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
        if self.client:
            self.disconnect()
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
