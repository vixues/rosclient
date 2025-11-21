#!/usr/bin/env python
"""Modern GUI test tool for RosClient using Pygame."""
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

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Point cloud display will be disabled.")


# Color scheme - Modern dark theme with refined colors
COLORS = {
    'bg': (15, 15, 20),
    'bg_secondary': (20, 20, 26),
    'surface': (30, 30, 38),
    'surface_light': (40, 40, 48),
    'surface_hover': (45, 45, 55),
    'primary': (100, 181, 246),
    'primary_dark': (66, 165, 245),
    'primary_light': (129, 212, 250),
    'success': (76, 175, 80),
    'success_light': (129, 199, 132),
    'warning': (255, 193, 7),
    'warning_light': (255, 224, 130),
    'error': (244, 67, 54),
    'error_light': (239, 154, 154),
    'text': (255, 255, 255),
    'text_secondary': (180, 180, 180),
    'text_tertiary': (120, 120, 120),
    'border': (50, 50, 60),
    'border_light': (70, 70, 80),
    'accent': (156, 39, 176),
    'accent_light': (186, 104, 200),
    'shadow': (0, 0, 0, 120),
    'shadow_light': (0, 0, 0, 60),
}

# Card design constants
CARD_RADIUS = 12
CARD_SHADOW_OFFSET = 4
CARD_BORDER_WIDTH = 1

# Spacing constants for consistent padding and margins
PADDING_SMALL = 10
PADDING_MEDIUM = 15
PADDING_LARGE = 20
SPACING_XS = 5
SPACING_SMALL = 10
SPACING_MEDIUM = 15
SPACING_LARGE = 20
SPACING_XL = 30


class Card:
    """Unified card component with consistent styling."""
    
    @staticmethod
    def draw(surface, rect, color=COLORS['surface'], border_color=COLORS['border'], 
             shadow=True, border_width=CARD_BORDER_WIDTH, radius=CARD_RADIUS):
        """Draw a card with shadow and border."""
        if shadow:
            # Draw shadow
            shadow_rect = rect.copy()
            shadow_rect.x += CARD_SHADOW_OFFSET
            shadow_rect.y += CARD_SHADOW_OFFSET
            shadow_surface = pygame.Surface((shadow_rect.width, shadow_rect.height), pygame.SRCALPHA)
            pygame.draw.rect(shadow_surface, COLORS['shadow'], shadow_surface.get_rect(), border_radius=radius)
            surface.blit(shadow_surface, shadow_rect)
        
        # Draw card background
        pygame.draw.rect(surface, color, rect, border_radius=radius)
        
        # Draw border
        if border_width > 0:
            pygame.draw.rect(surface, border_color, rect, width=border_width, border_radius=radius)
    
    @staticmethod
    def draw_header(surface, rect, title, font, title_color=COLORS['text'], 
                   bg_color=COLORS['surface_light'], border_color=COLORS['border']):
        """Draw a card header section with proper padding."""
        header_rect = pygame.Rect(rect.x, rect.y, rect.width, 45)
        pygame.draw.rect(surface, bg_color, header_rect, border_radius=CARD_RADIUS)
        pygame.draw.rect(surface, border_color, header_rect, width=1, border_radius=CARD_RADIUS)
        
        # Draw title with proper padding
        title_surface = font.render(title, True, title_color)
        title_y = rect.y + (header_rect.height - title_surface.get_height()) // 2
        surface.blit(title_surface, (rect.x + PADDING_MEDIUM, title_y))
        
        return header_rect


class ModernButton:
    """Modern styled button component with animations."""
    
    def __init__(self, x, y, width, height, text, callback=None, color=COLORS['primary']):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.callback = callback
        self.color = color
        self.hover_color = tuple(min(255, c + 30) for c in color)
        self.pressed = False
        self.hovered = False
        self.animation_scale = 1.0
        self.target_scale = 1.0
        
    def handle_event(self, event):
        """Handle pygame events."""
        if event.type == pygame.MOUSEMOTION:
            was_hovered = self.hovered
            self.hovered = self.rect.collidepoint(event.pos)
            if self.hovered and not was_hovered:
                self.target_scale = 1.05
            elif not self.hovered:
                self.target_scale = 1.0
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.pressed = True
                self.target_scale = 0.95
        elif event.type == pygame.MOUSEBUTTONUP:
            if self.pressed and self.rect.collidepoint(event.pos):
                if self.callback:
                    self.callback()
            self.pressed = False
            self.target_scale = 1.05 if self.hovered else 1.0
            
    def update(self, dt):
        """Update animation."""
        if abs(self.animation_scale - self.target_scale) > 0.01:
            diff = self.target_scale - self.animation_scale
            self.animation_scale += diff * dt * 10
            
    def draw(self, surface, font):
        """Draw the button with refined card-style design."""
        # Animated color - ensure all values are integers in 0-255 range
        if self.hovered:
            color = tuple(max(0, min(255, int(c * (0.7 + 0.3 * self.animation_scale)))) for c in self.hover_color)
        else:
            color = self.color
        if self.pressed:
            color = tuple(max(0, min(255, c - 40)) for c in color)
        
        # Ensure color is a tuple of integers
        color = tuple(int(max(0, min(255, c))) for c in color)
            
        # Calculate scaled rect
        scale = self.animation_scale
        scaled_rect = pygame.Rect(
            self.rect.centerx - self.rect.width * scale / 2,
            self.rect.centery - self.rect.height * scale / 2,
            self.rect.width * scale,
            self.rect.height * scale
        )
        
        # Draw button with card-style shadow
        shadow_rect = scaled_rect.copy()
        shadow_rect.x += 3
        shadow_rect.y += 3
        shadow_surface = pygame.Surface((shadow_rect.width, shadow_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, COLORS['shadow_light'], shadow_surface.get_rect(), border_radius=CARD_RADIUS)
        surface.blit(shadow_surface, shadow_rect)
        
        # Draw button with refined rounded corners
        pygame.draw.rect(surface, color, scaled_rect, border_radius=CARD_RADIUS)
        
        # Draw subtle border with gradient effect
        border_color = tuple(max(0, min(255, int(c * 1.2))) for c in color)
        pygame.draw.rect(surface, border_color, scaled_rect, width=1, border_radius=CARD_RADIUS)
        
        # Draw text with better positioning
        text_surface = font.render(self.text, True, COLORS['text'])
        text_rect = text_surface.get_rect(center=scaled_rect.center)
        # Add subtle text shadow for better readability
        shadow_text = font.render(self.text, True, (0, 0, 0, 100))
        surface.blit(shadow_text, (text_rect.x + 1, text_rect.y + 1))
        surface.blit(text_surface, text_rect)


class ModernInput:
    """Modern styled input field component."""
    
    def __init__(self, x, y, width, height, default_text="", placeholder="", multiline=False):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = default_text
        self.placeholder = placeholder
        self.active = False
        self.cursor_visible = True
        self.cursor_timer = 0
        self.multiline = multiline
        self.scroll_y = 0
        self.cursor_pos = len(self.text) if not multiline else [0, 0]  # [line, col] for multiline
        
    def handle_event(self, event):
        """Handle pygame events."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
            if event.button == 4:  # Scroll up
                self.scroll_y = max(0, self.scroll_y - 20)
            elif event.button == 5:  # Scroll down
                self.scroll_y += 20
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                if self.multiline:
                    lines = self.text.split('\n')
                    if self.cursor_pos[0] < len(lines):
                        if self.cursor_pos[1] > 0:
                            lines[self.cursor_pos[0]] = lines[self.cursor_pos[0]][:self.cursor_pos[1]-1] + lines[self.cursor_pos[0]][self.cursor_pos[1]:]
                            self.cursor_pos[1] -= 1
                        elif self.cursor_pos[0] > 0:
                            self.cursor_pos[1] = len(lines[self.cursor_pos[0]-1])
                            lines[self.cursor_pos[0]-1] += lines.pop(self.cursor_pos[0])
                            self.cursor_pos[0] -= 1
                        self.text = '\n'.join(lines)
                else:
                    self.text = self.text[:-1]
            elif event.key == pygame.K_RETURN:
                if self.multiline:
                    lines = self.text.split('\n')
                    if self.cursor_pos[0] < len(lines):
                        line = lines[self.cursor_pos[0]]
                        lines[self.cursor_pos[0]] = line[:self.cursor_pos[1]]
                        lines.insert(self.cursor_pos[0] + 1, line[self.cursor_pos[1]:])
                        self.cursor_pos[0] += 1
                        self.cursor_pos[1] = 0
                        self.text = '\n'.join(lines)
                else:
                    self.active = False
            elif event.key == pygame.K_UP and self.multiline:
                self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif event.key == pygame.K_DOWN and self.multiline:
                lines = self.text.split('\n')
                self.cursor_pos[0] = min(len(lines) - 1, self.cursor_pos[0] + 1)
            elif event.key == pygame.K_LEFT:
                if self.multiline:
                    lines = self.text.split('\n')
                    if self.cursor_pos[1] > 0:
                        self.cursor_pos[1] -= 1
                    elif self.cursor_pos[0] > 0:
                        self.cursor_pos[0] -= 1
                        self.cursor_pos[1] = len(lines[self.cursor_pos[0]])
                else:
                    pass  # Single line cursor handled by text
            elif event.key == pygame.K_RIGHT:
                if self.multiline:
                    lines = self.text.split('\n')
                    if self.cursor_pos[0] < len(lines):
                        if self.cursor_pos[1] < len(lines[self.cursor_pos[0]]):
                            self.cursor_pos[1] += 1
                        elif self.cursor_pos[0] < len(lines) - 1:
                            self.cursor_pos[0] += 1
                            self.cursor_pos[1] = 0
                else:
                    pass
            else:
                if self.multiline:
                    lines = self.text.split('\n')
                    if self.cursor_pos[0] < len(lines):
                        line = lines[self.cursor_pos[0]]
                        lines[self.cursor_pos[0]] = line[:self.cursor_pos[1]] + event.unicode + line[self.cursor_pos[1]:]
                        self.cursor_pos[1] += 1
                        self.text = '\n'.join(lines)
                else:
                    self.text += event.unicode
                
    def update(self, dt):
        """Update cursor blink."""
        self.cursor_timer += dt
        if self.cursor_timer > 0.5:
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = 0
            
    def draw(self, surface, font):
        """Draw the input field with refined card-style design."""
        # Draw card-style background
        bg_color = COLORS['surface_light'] if self.active else COLORS['surface']
        border_color = COLORS['primary'] if self.active else COLORS['border']
        
        # Draw shadow
        shadow_rect = self.rect.copy()
        shadow_rect.x += 2
        shadow_rect.y += 2
        shadow_surface = pygame.Surface((shadow_rect.width, shadow_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, COLORS['shadow_light'], shadow_surface.get_rect(), border_radius=CARD_RADIUS)
        surface.blit(shadow_surface, shadow_rect)
        
        # Draw background
        pygame.draw.rect(surface, bg_color, self.rect, border_radius=CARD_RADIUS)
        pygame.draw.rect(surface, border_color, self.rect, width=1 if self.active else CARD_BORDER_WIDTH, border_radius=CARD_RADIUS)
        
        # Create clipping area
        clip_rect = self.rect.inflate(-10, -10)
        old_clip = surface.get_clip()
        surface.set_clip(clip_rect)
        
        if self.multiline:
            # Draw multiline text
            lines = self.text.split('\n') if self.text else []
            if not lines and self.placeholder:
                placeholder_surface = font.render(self.placeholder, True, COLORS['text_secondary'])
                surface.blit(placeholder_surface, (clip_rect.x, clip_rect.y - self.scroll_y))
            else:
                y_offset = clip_rect.y - self.scroll_y
                for i, line in enumerate(lines):
                    if y_offset + i * (font.get_height() + 2) > clip_rect.bottom:
                        break
                    if y_offset + i * (font.get_height() + 2) + font.get_height() < clip_rect.y:
                        continue
                    line_surface = font.render(line or " ", True, COLORS['text'])
                    surface.blit(line_surface, (clip_rect.x, y_offset + i * (font.get_height() + 2)))
                
                # Draw cursor for multiline
                if self.active and self.cursor_visible:
                    cursor_y = clip_rect.y - self.scroll_y + self.cursor_pos[0] * (font.get_height() + 2)
                    if clip_rect.y <= cursor_y <= clip_rect.bottom:
                        lines = self.text.split('\n')
                        cursor_x = clip_rect.x
                        if self.cursor_pos[0] < len(lines):
                            cursor_text = lines[self.cursor_pos[0]][:self.cursor_pos[1]]
                            if cursor_text:
                                cursor_surface = font.render(cursor_text, True, COLORS['text'])
                                cursor_x = clip_rect.x + cursor_surface.get_width()
                        pygame.draw.line(surface, COLORS['text'], 
                                       (cursor_x, cursor_y),
                                       (cursor_x, cursor_y + font.get_height()), 2)
        else:
            # Draw single line text
            display_text = self.text if self.text else self.placeholder
            text_color = COLORS['text'] if self.text else COLORS['text_secondary']
            
            # Truncate if too long
            text_surface = font.render(display_text, True, text_color)
            if text_surface.get_width() > clip_rect.width:
                # Find fitting text
                for i in range(len(display_text), 0, -1):
                    truncated = display_text[-i:] if i < len(display_text) else display_text
                    test_surface = font.render(truncated, True, text_color)
                    if test_surface.get_width() <= clip_rect.width:
                        text_surface = test_surface
                        break
                else:
                    text_surface = font.render("...", True, text_color)
            
            surface.blit(text_surface, (clip_rect.x, clip_rect.centery - font.get_height() // 2))
            
            # Draw cursor for single line
            if self.active and self.cursor_visible:
                cursor_x = clip_rect.x + text_surface.get_width() + 2
                pygame.draw.line(surface, COLORS['text'], 
                               (cursor_x, clip_rect.y + 5),
                               (cursor_x, clip_rect.bottom - 5), 2)
        
        surface.set_clip(old_clip)


class ModernCheckbox:
    """Modern styled checkbox component."""
    
    def __init__(self, x, y, text, checked=False, callback=None):
        self.rect = pygame.Rect(x, y, 20, 20)
        self.text = text
        self.checked = checked
        self.callback = callback
        
    def handle_event(self, event):
        """Handle pygame events."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.checked = not self.checked
                if self.callback:
                    self.callback(self.checked)
                    
    def draw(self, surface, font):
        """Draw the checkbox with refined design."""
        # Draw box with card-style
        bg_color = COLORS['primary'] if self.checked else COLORS['surface_light']
        pygame.draw.rect(surface, bg_color, self.rect, border_radius=6)
        border_color = COLORS['primary_light'] if self.checked else COLORS['border']
        pygame.draw.rect(surface, border_color, self.rect, width=1, border_radius=6)
        
        # Draw checkmark with smooth lines
        if self.checked:
            # Draw a more refined checkmark
            check_color = COLORS['text']
            points = [
                (self.rect.x + 5, self.rect.y + 10),
                (self.rect.x + 9, self.rect.y + 14),
                (self.rect.x + 15, self.rect.y + 6)
            ]
            pygame.draw.lines(surface, check_color, False, points, 3)
            # Add a subtle glow
            pygame.draw.lines(surface, COLORS['primary_light'], False, points, 1)
            
        # Draw text
        text_surface = font.render(self.text, True, COLORS['text'])
        text_rect = text_surface.get_rect(midleft=(self.rect.right + 12, self.rect.centery))
        surface.blit(text_surface, text_rect)


class JSONEditor:
    """Professional JSON editor with syntax highlighting and line numbers."""
    
    def __init__(self, x, y, width, height, default_text=""):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = default_text
        self.active = False
        self.cursor_visible = True
        self.cursor_timer = 0
        self.scroll_y = 0
        self.scroll_x = 0
        self.cursor_pos = [0, 0]  # [line, col]
        self.line_height = 20
        self.char_width = 8
        self.line_number_width = 50
        
    def handle_event(self, event):
        """Handle pygame events."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
            if event.button == 4:  # Scroll up
                self.scroll_y = max(0, self.scroll_y - 20)
            elif event.button == 5:  # Scroll down
                self.scroll_y += 20
        elif event.type == pygame.KEYDOWN and self.active:
            lines = self.text.split('\n')
            if event.key == pygame.K_BACKSPACE:
                if self.cursor_pos[0] < len(lines):
                    if self.cursor_pos[1] > 0:
                        lines[self.cursor_pos[0]] = lines[self.cursor_pos[0]][:self.cursor_pos[1]-1] + lines[self.cursor_pos[0]][self.cursor_pos[1]:]
                        self.cursor_pos[1] -= 1
                    elif self.cursor_pos[0] > 0:
                        self.cursor_pos[1] = len(lines[self.cursor_pos[0]-1])
                        lines[self.cursor_pos[0]-1] += lines.pop(self.cursor_pos[0])
                        self.cursor_pos[0] -= 1
                    self.text = '\n'.join(lines)
            elif event.key == pygame.K_RETURN:
                if self.cursor_pos[0] < len(lines):
                    line = lines[self.cursor_pos[0]]
                    lines[self.cursor_pos[0]] = line[:self.cursor_pos[1]]
                    lines.insert(self.cursor_pos[0] + 1, line[self.cursor_pos[1]:])
                    self.cursor_pos[0] += 1
                    self.cursor_pos[1] = 0
                    self.text = '\n'.join(lines)
            elif event.key == pygame.K_TAB:
                # Insert 4 spaces for indentation
                if self.cursor_pos[0] < len(lines):
                    lines[self.cursor_pos[0]] = lines[self.cursor_pos[0]][:self.cursor_pos[1]] + "    " + lines[self.cursor_pos[0]][self.cursor_pos[1]:]
                    self.cursor_pos[1] += 4
                    self.text = '\n'.join(lines)
            elif event.key == pygame.K_UP:
                self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
                self.cursor_pos[1] = min(self.cursor_pos[1], len(lines[self.cursor_pos[0]]))
            elif event.key == pygame.K_DOWN:
                self.cursor_pos[0] = min(len(lines) - 1, self.cursor_pos[0] + 1)
                self.cursor_pos[1] = min(self.cursor_pos[1], len(lines[self.cursor_pos[0]]))
            elif event.key == pygame.K_LEFT:
                if self.cursor_pos[1] > 0:
                    self.cursor_pos[1] -= 1
                elif self.cursor_pos[0] > 0:
                    self.cursor_pos[0] -= 1
                    self.cursor_pos[1] = len(lines[self.cursor_pos[0]])
            elif event.key == pygame.K_RIGHT:
                if self.cursor_pos[0] < len(lines):
                    if self.cursor_pos[1] < len(lines[self.cursor_pos[0]]):
                        self.cursor_pos[1] += 1
                    elif self.cursor_pos[0] < len(lines) - 1:
                        self.cursor_pos[0] += 1
                        self.cursor_pos[1] = 0
            else:
                if self.cursor_pos[0] < len(lines):
                    line = lines[self.cursor_pos[0]]
                    lines[self.cursor_pos[0]] = line[:self.cursor_pos[1]] + event.unicode + line[self.cursor_pos[1]:]
                    self.cursor_pos[1] += 1
                    self.text = '\n'.join(lines)
                    
    def update(self, dt):
        """Update cursor blink."""
        self.cursor_timer += dt
        if self.cursor_timer > 0.5:
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = 0
            
    def format_json(self):
        """Format JSON text."""
        try:
            obj = json.loads(self.text)
            self.text = json.dumps(obj, indent=4)
            lines = self.text.split('\n')
            self.cursor_pos = [0, 0]
        except:
            pass
            
    def draw(self, surface, font):
        """Draw the JSON editor with refined card-style design."""
        # Draw card-style background
        bg_color = COLORS['surface_light'] if self.active else COLORS['surface']
        border_color = COLORS['primary'] if self.active else COLORS['border']
        
        # Draw shadow
        shadow_rect = self.rect.copy()
        shadow_rect.x += CARD_SHADOW_OFFSET
        shadow_rect.y += CARD_SHADOW_OFFSET
        shadow_surface = pygame.Surface((shadow_rect.width, shadow_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, COLORS['shadow'], shadow_surface.get_rect(), border_radius=CARD_RADIUS)
        surface.blit(shadow_surface, shadow_rect)
        
        # Draw background
        pygame.draw.rect(surface, bg_color, self.rect, border_radius=CARD_RADIUS)
        pygame.draw.rect(surface, border_color, self.rect, width=1 if self.active else CARD_BORDER_WIDTH, border_radius=CARD_RADIUS)
        
        # Draw line numbers area
        line_num_rect = pygame.Rect(self.rect.x + 5, self.rect.y + 5, self.line_number_width, self.rect.height - 10)
        pygame.draw.rect(surface, COLORS['bg'], line_num_rect)
        
        # Create clipping area for text
        clip_rect = pygame.Rect(
            self.rect.x + self.line_number_width + 10,
            self.rect.y + 5,
            self.rect.width - self.line_number_width - 15,
            self.rect.height - 10
        )
        old_clip = surface.get_clip()
        surface.set_clip(clip_rect)
        
        lines = self.text.split('\n') if self.text else [""]
        y_offset = clip_rect.y - self.scroll_y
        
        for i, line in enumerate(lines):
            line_y = y_offset + i * self.line_height
            if line_y + self.line_height < clip_rect.y:
                continue
            if line_y > clip_rect.bottom:
                break
                
            # Draw line number
            line_num_surface = font.render(str(i + 1), True, COLORS['text_secondary'])
            surface.blit(line_num_surface, (self.rect.x + 10, line_y))
            
            # Draw line with syntax highlighting
            x_pos = clip_rect.x - self.scroll_x
            for j, char in enumerate(line):
                char_x = x_pos + j * self.char_width
                if char_x + self.char_width < clip_rect.x:
                    continue
                if char_x > clip_rect.right:
                    break
                    
                # Simple syntax highlighting
                if char in ['{', '}', '[', ']']:
                    color = COLORS['primary']
                elif char in [':', ',']:
                    color = COLORS['text_secondary']
                elif line.strip().startswith('"') and j < len(line) - 1 and line[j+1] == '"':
                    color = COLORS['success']
                else:
                    color = COLORS['text']
                    
                char_surface = font.render(char, True, color)
                surface.blit(char_surface, (char_x, line_y))
        
        # Draw cursor
        if self.active and self.cursor_visible:
            cursor_y = clip_rect.y - self.scroll_y + self.cursor_pos[0] * self.line_height
            if clip_rect.y <= cursor_y <= clip_rect.bottom:
                lines = self.text.split('\n')
                cursor_x = clip_rect.x - self.scroll_x + self.cursor_pos[1] * self.char_width
                if self.cursor_pos[0] < len(lines):
                    pygame.draw.line(surface, COLORS['text'], 
                                   (cursor_x, cursor_y),
                                   (cursor_x, cursor_y + self.line_height), 2)
        
        surface.set_clip(old_clip)


class TopicList:
    """Topic list display component with name and type."""
    
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.topics = []  # List of tuples: (name, type)
        self.selected_index = -1
        self.scroll_y = 0
        
    def set_topics(self, topics):
        """Set the list of topics. Can be list of strings or list of (name, type) tuples."""
        if topics and isinstance(topics[0], tuple):
            self.topics = topics
        else:
            # Convert list of strings to list of tuples with empty type
            self.topics = [(topic, "") if isinstance(topic, str) else topic for topic in topics]
        
    def handle_event(self, event):
        """Handle pygame events. Returns (name, type) tuple if topic selected."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                rel_y = event.pos[1] - self.rect.y - 5
                index = (rel_y + self.scroll_y) // 30
                if 0 <= index < len(self.topics):
                    self.selected_index = index
                    return self.topics[index]
            if event.button == 4:  # Scroll up
                self.scroll_y = max(0, self.scroll_y - 20)
            elif event.button == 5:  # Scroll down
                self.scroll_y += 20
        return None
        
    def draw(self, surface, font, small_font):
        """Draw the topic list with refined card-style design."""
        # Draw card with shadow
        Card.draw(surface, self.rect, COLORS['surface'], COLORS['border'], shadow=True, radius=CARD_RADIUS)
        
        # Draw header with card header style
        header_rect = Card.draw_header(surface, self.rect, "Topics", small_font, 
                                       COLORS['text'], COLORS['surface_light'], COLORS['border'])
        
        # Draw topics with proper spacing
        header_height = 45
        y_offset = self.rect.y + header_height + PADDING_SMALL - self.scroll_y
        item_height = 35
        
        for i, topic_data in enumerate(self.topics):
            topic_y = y_offset + i * (item_height + SPACING_XS)
            if topic_y + item_height < self.rect.y + header_height:
                continue
            if topic_y > self.rect.bottom:
                break
                
            topic_name = topic_data[0] if isinstance(topic_data, tuple) else topic_data
            topic_type = topic_data[1] if isinstance(topic_data, tuple) and len(topic_data) > 1 else ""
            
            # Highlight selected with refined style
            if i == self.selected_index:
                highlight_rect = pygame.Rect(self.rect.x + PADDING_SMALL, topic_y, 
                                           self.rect.width - PADDING_SMALL * 2, item_height)
                pygame.draw.rect(surface, COLORS['primary'], highlight_rect, border_radius=6)
                pygame.draw.rect(surface, COLORS['primary_light'], highlight_rect, width=1, border_radius=6)
                
            # Draw topic name with proper padding
            name_color = COLORS['text'] if i == self.selected_index else COLORS['text']
            topic_surface = small_font.render(topic_name, True, name_color)
            name_y = topic_y + (item_height - topic_surface.get_height()) // 2
            if not topic_type:
                name_y = topic_y + (item_height - topic_surface.get_height()) // 2
            surface.blit(topic_surface, (self.rect.x + PADDING_MEDIUM, name_y))
            
            # Draw topic type (if available) with proper spacing
            if topic_type:
                type_surface = small_font.render(topic_type, True, COLORS['text_secondary'])
                type_y = topic_y + topic_surface.get_height() + SPACING_XS
                if type_y + type_surface.get_height() < self.rect.bottom:
                    surface.blit(type_surface, (self.rect.x + PADDING_MEDIUM, type_y))


class TabBar:
    """Tab navigation bar."""
    
    def __init__(self, x, y, width, height, tabs):
        self.rect = pygame.Rect(x, y, width, height)
        self.tabs = tabs
        self.active_tab = 0
        self.tab_width = width // len(tabs)
        
    def handle_event(self, event):
        """Handle pygame events."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, tab in enumerate(self.tabs):
                tab_rect = pygame.Rect(
                    self.rect.x + i * self.tab_width,
                    self.rect.y,
                    self.tab_width,
                    self.rect.height
                )
                if tab_rect.collidepoint(event.pos):
                    self.active_tab = i
                    return i
        return None
        
    def draw(self, surface, font):
        """Draw the tab bar with refined design."""
        # Draw background with subtle border
        pygame.draw.rect(surface, COLORS['bg_secondary'], self.rect)
        pygame.draw.line(surface, COLORS['border'], (self.rect.x, self.rect.bottom - 1), 
                        (self.rect.right, self.rect.bottom - 1), 1)
        
        # Draw tabs
        for i, tab in enumerate(self.tabs):
            tab_rect = pygame.Rect(
                self.rect.x + i * self.tab_width,
                self.rect.y,
                self.tab_width,
                self.rect.height
            )
            
            # Active tab with card-style
            if i == self.active_tab:
                # Draw active tab background
                active_bg_rect = pygame.Rect(tab_rect.x + 2, tab_rect.y + 2, 
                                            tab_rect.width - 4, tab_rect.height - 2)
                pygame.draw.rect(surface, COLORS['surface'], active_bg_rect, border_radius=8)
                # Draw indicator line
                pygame.draw.line(surface, COLORS['primary'], 
                               (tab_rect.x, tab_rect.bottom - 2),
                               (tab_rect.right, tab_rect.bottom - 2), 3)
                text_color = COLORS['text']
            else:
                text_color = COLORS['text_secondary']
            
            # Tab text
            text_surface = font.render(tab, True, text_color)
            text_rect = text_surface.get_rect(center=tab_rect.center)
            surface.blit(text_surface, text_rect)


class RosClientPygameGUI:
    """Modern Pygame GUI for RosClient."""
    
    def __init__(self):
        pygame.init()
        self.screen_width = 1400
        self.screen_height = 900
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("ROS Client - Modern GUI")
        
        self.clock = pygame.time.Clock()
        self.running = True
        self.dt = 0
        
        # Fonts
        self.font_large = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # Client state
        self.client: Optional[RosClient] = None
        self.is_connected = False
        self.update_thread: Optional[threading.Thread] = None
        self.stop_update = threading.Event()
        self.image_queue = queue.Queue(maxsize=1)
        self.current_image = None
        
        # Tabs
        self.tabs = ["Connection", "Status", "Image", "Control", "Point Cloud", "Network Test"]
        self.tab_bar = TabBar(0, 0, self.screen_width, 50, self.tabs)
        self.current_tab = 0
        
        # UI Components
        self.setup_ui()
        self.setup_update_loop()
        
    def setup_ui(self):
        """Setup UI components."""
        # Connection tab components
        self.connection_url_input = ModernInput(200, 100, 400, 35, "ws://localhost:9090")
        self.use_mock_checkbox = ModernCheckbox(200, 150, "Use Mock Client (Test Mode)", False)
        self.connect_btn = ModernButton(200, 200, 120, 40, "Connect", self.connect)
        self.disconnect_btn = ModernButton(330, 200, 120, 40, "Disconnect", self.disconnect)
        self.disconnect_btn.color = COLORS['error']
        self.disconnect_btn.hover_color = (min(255, COLORS['error'][0] + 20), 
                                          min(255, COLORS['error'][1] + 20),
                                          min(255, COLORS['error'][2] + 20))
        
        # Status display
        self.status_labels = {}
        self.status_values = {}
        
        # Control tab components
        self.topic_list = TopicList(50, 100, 300, 500)
        self.control_topic_input = ModernInput(370, 100, 380, 40, "/control")
        self.control_type_input = ModernInput(770, 100, 380, 40, "controller_msgs/cmd")
        self.json_editor = JSONEditor(370, 160, 780, 340, '{\n    "cmd": 1\n}')
        self.command_history = []
        
        # Point cloud components
        self.current_point_cloud = None
        self.pc_surface = None
        
        # Network test components
        self.test_url_input = ModernInput(200, 100, 400, 35, "ws://localhost:9090")
        self.test_timeout_input = ModernInput(200, 150, 100, 35, "5")
        self.test_results = []
        self.test_btn = ModernButton(200, 200, 150, 40, "Test Connection", self.test_connection, COLORS['primary'])
        
        # Preset command buttons (moved outside text box)
        self.preset_buttons = [
            ModernButton(370 + i * 140, 520, 130, 38, name, 
                        lambda c=cmd: self.set_preset_command(c))
            for i, (name, cmd) in enumerate([
                ("Takeoff", '{\n    "cmd": 1\n}'),
                ("Land", '{\n    "cmd": 2\n}'),
                ("Return", '{\n    "cmd": 3\n}'),
                ("Hover", '{\n    "cmd": 4\n}'),
            ])
        ]
        self.format_json_btn = ModernButton(370, 570, 140, 42, "Format JSON", 
                                           self.format_json, COLORS['accent'])
        self.send_command_btn = ModernButton(520, 570, 160, 42, "Send Command", 
                                            self.send_control_command,
                                            COLORS['success'])
        
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
                        if self.current_tab == 2 and HAS_CV2:  # Image tab
                            self.update_image()
                        if self.current_tab == 3:  # Control tab - update topic list
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
                    # Update topic list after connection
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
            self.pc_surface = pygame.image.frombuffer(buf, size, "RGBA")
            fig.clear()
        except Exception as e:
            print(f"Point cloud render error: {e}")
            
    def set_preset_command(self, command: str):
        """Set preset command."""
        self.json_editor.text = command
        self.json_editor.cursor_pos = [0, 0]
        
    def format_json(self):
        """Format JSON in editor."""
        self.json_editor.format_json()
        
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
            self.topic_list.set_topics(topics)
        except Exception as e:
            # Fallback to empty list
            self.topic_list.set_topics([])
        
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
        
    def draw_connection_tab(self):
        """Draw connection configuration tab with refined card design and proper spacing."""
        y = 70
        
        # Title with card-style
        title_card = pygame.Rect(50, y, 500, 50)
        Card.draw(self.screen, title_card, COLORS['surface'], COLORS['border'], shadow=True)
        title = self.font_large.render("Connection Configuration", True, COLORS['text'])
        title_y = title_card.y + (title_card.height - title.get_height()) // 2
        self.screen.blit(title, (title_card.x + PADDING_MEDIUM, title_y))
        y += title_card.height + SPACING_LARGE
        
        # Connection settings card
        settings_card = pygame.Rect(50, y, self.screen_width - 100, 220)
        Card.draw(self.screen, settings_card, COLORS['surface'], COLORS['border'], shadow=True)
        header_rect = Card.draw_header(self.screen, settings_card, "Connection Settings", self.font_medium)
        
        card_y = settings_card.y + header_rect.height + SPACING_MEDIUM
        
        # Connection URL with proper spacing
        label = self.font_medium.render("WebSocket Address:", True, COLORS['text'])
        self.screen.blit(label, (settings_card.x + PADDING_MEDIUM, card_y))
        self.connection_url_input.rect.x = settings_card.x + PADDING_MEDIUM
        self.connection_url_input.rect.y = card_y + label.get_height() + SPACING_SMALL
        self.connection_url_input.rect.width = settings_card.width - PADDING_MEDIUM * 2
        self.connection_url_input.rect.height = 35
        self.connection_url_input.draw(self.screen, self.font_medium)
        card_y += label.get_height() + SPACING_SMALL + self.connection_url_input.rect.height + SPACING_MEDIUM
        
        # Mock checkbox with proper spacing
        self.use_mock_checkbox.rect.x = settings_card.x + PADDING_MEDIUM
        self.use_mock_checkbox.rect.y = card_y
        self.use_mock_checkbox.draw(self.screen, self.font_medium)
        card_y += self.use_mock_checkbox.rect.height + SPACING_MEDIUM
        
        # Buttons with proper spacing
        self.connect_btn.rect.x = settings_card.x + PADDING_MEDIUM
        self.connect_btn.rect.y = card_y
        self.connect_btn.rect.width = 130
        self.connect_btn.rect.height = 40
        self.connect_btn.draw(self.screen, self.font_medium)
        self.disconnect_btn.rect.x = settings_card.x + PADDING_MEDIUM + self.connect_btn.rect.width + SPACING_MEDIUM
        self.disconnect_btn.rect.y = card_y
        self.disconnect_btn.rect.width = 130
        self.disconnect_btn.rect.height = 40
        self.disconnect_btn.draw(self.screen, self.font_medium)
        
        y += settings_card.height + SPACING_LARGE
        
        # Log display card
        log_card = pygame.Rect(50, y, self.screen_width - 100, self.screen_height - y - SPACING_MEDIUM)
        Card.draw(self.screen, log_card, COLORS['surface'], COLORS['border'], shadow=True)
        header_rect = Card.draw_header(self.screen, log_card, "Connection Log", self.font_medium)
        
        log_area = pygame.Rect(log_card.x + PADDING_MEDIUM, log_card.y + header_rect.height + PADDING_MEDIUM, 
                               log_card.width - PADDING_MEDIUM * 2, 
                               log_card.height - header_rect.height - PADDING_MEDIUM * 2)
        pygame.draw.rect(self.screen, COLORS['bg'], log_area, border_radius=8)
        
        if hasattr(self, 'connection_logs'):
            log_y = log_area.y + PADDING_SMALL
            for log in self.connection_logs[-20:]:  # Show last 20 logs
                log_surface = self.font_small.render(log, True, COLORS['text_secondary'])
                if log_y + log_surface.get_height() > log_area.bottom - PADDING_SMALL:
                    break
                self.screen.blit(log_surface, (log_area.x + PADDING_MEDIUM, log_y))
                log_y += log_surface.get_height() + SPACING_XS
                    
    def draw_status_tab(self):
        """Draw status monitoring tab with refined card design and proper spacing."""
        y = 70
        
        # Title card with connection indicator
        title_card = pygame.Rect(50, y, self.screen_width - 100, 50)
        Card.draw(self.screen, title_card, COLORS['surface'], COLORS['border'], shadow=True)
        title = self.font_large.render("Status Monitoring", True, COLORS['text'])
        title_y = title_card.y + (title_card.height - title.get_height()) // 2
        self.screen.blit(title, (title_card.x + PADDING_MEDIUM, title_y))
        
        # Connection indicator badge
        if self.is_connected:
            indicator_color = COLORS['success']
            indicator_text = "● Connected"
        else:
            indicator_color = COLORS['error']
            indicator_text = "● Disconnected"
        indicator = self.font_medium.render(indicator_text, True, indicator_color)
        indicator_rect = indicator.get_rect(midright=(title_card.right - PADDING_MEDIUM, title_card.centery))
        self.screen.blit(indicator, indicator_rect)
        y += title_card.height + SPACING_LARGE
        
        # Status fields with refined cards
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
        
        x_start = 50
        x_offset = (self.screen_width - 100) // 2
        card_width = x_offset - SPACING_MEDIUM
        card_height = 45
        
        for i, (label, field) in enumerate(status_fields):
            x = x_start if i % 2 == 0 else x_start + x_offset
            row = i // 2
            card_y = y + row * (card_height + SPACING_SMALL)
            
            # Draw refined card
            card_rect = pygame.Rect(x, card_y, card_width, card_height)
            Card.draw(self.screen, card_rect, COLORS['surface'], COLORS['border'], shadow=False)
            
            # Label with proper padding
            label_surface = self.font_medium.render(f"{label}:", True, COLORS['text_secondary'])
            label_y = card_y + (card_height - label_surface.get_height()) // 2
            self.screen.blit(label_surface, (x + PADDING_MEDIUM, label_y))
            
            # Value with color coding
            value = self.status_values.get(field, "N/A")
            if field == "connected":
                value_color = COLORS['success'] if value == "Connected" else COLORS['error']
            elif field == "armed":
                value_color = COLORS['warning'] if value == "Armed" else COLORS['text_secondary']
            elif field == "battery":
                try:
                    battery_val = float(value.replace('%', ''))
                    if battery_val < 20:
                        value_color = COLORS['error']
                    elif battery_val < 50:
                        value_color = COLORS['warning']
                    else:
                        value_color = COLORS['success']
                except:
                    value_color = COLORS['text']
            else:
                value_color = COLORS['text']
                
            value_surface = self.font_medium.render(str(value), True, value_color)
            value_x = x + card_width - value_surface.get_width() - PADDING_MEDIUM
            value_y = card_y + (card_height - value_surface.get_height()) // 2
            self.screen.blit(value_surface, (value_x, value_y))
            
    def draw_image_tab(self):
        """Draw image display tab with refined card design and proper spacing."""
        y = 70
        
        # Title card
        title_card = pygame.Rect(50, y, 500, 50)
        Card.draw(self.screen, title_card, COLORS['surface'], COLORS['border'], shadow=True)
        title = self.font_large.render("Image Display", True, COLORS['text'])
        title_y = title_card.y + (title_card.height - title.get_height()) // 2
        self.screen.blit(title, (title_card.x + PADDING_MEDIUM, title_y))
        y += title_card.height + SPACING_LARGE
        
        # Image display area with refined card style and proper padding
        img_area = pygame.Rect(50, y, self.screen_width - 100, self.screen_height - y - SPACING_MEDIUM)
        Card.draw(self.screen, img_area, COLORS['surface'], COLORS['border'], shadow=True)
        
        # Inner area with padding
        img_inner = pygame.Rect(img_area.x + PADDING_MEDIUM, img_area.y + PADDING_MEDIUM,
                                img_area.width - PADDING_MEDIUM * 2, img_area.height - PADDING_MEDIUM * 2)
        
        if self.current_image:
            # Center image
            img_rect = self.current_image.get_rect()
            img_rect.center = img_inner.center
            # Scale if needed
            if img_rect.width > img_inner.width or img_rect.height > img_inner.height:
                scale = min(img_inner.width / img_rect.width, img_inner.height / img_rect.height)
                new_size = (int(img_rect.width * scale), int(img_rect.height * scale))
                self.current_image = pygame.transform.scale(self.current_image, new_size)
                img_rect = self.current_image.get_rect(center=img_inner.center)
            self.screen.blit(self.current_image, img_rect)
        else:
            # Placeholder with refined card-style message
            placeholder_card = pygame.Rect(img_inner.centerx - 200, img_inner.centery - 50, 400, 100)
            Card.draw(self.screen, placeholder_card, COLORS['surface_light'], COLORS['border'], shadow=False)
            placeholder_text = self.font_medium.render("Waiting for image data...", True, COLORS['text_secondary'])
            placeholder_rect = placeholder_text.get_rect(center=(placeholder_card.centerx, placeholder_card.centery - 10))
            self.screen.blit(placeholder_text, placeholder_rect)
            
            # Loading animation
            if self.is_connected:
                loading_text = self.font_small.render("Fetching image...", True, COLORS['primary'])
                loading_rect = loading_text.get_rect(center=(placeholder_card.centerx, placeholder_card.centery + 15))
                self.screen.blit(loading_text, loading_rect)
            
    def draw_control_tab(self):
        """Draw control command tab with refined card design and proper spacing."""
        y = 70
        
        # Title card
        title_card = pygame.Rect(50, y, 500, 50)
        Card.draw(self.screen, title_card, COLORS['surface'], COLORS['border'], shadow=True)
        title = self.font_large.render("Control Commands", True, COLORS['text'])
        title_y = title_card.y + (title_card.height - title.get_height()) // 2
        self.screen.blit(title, (title_card.x + PADDING_MEDIUM, title_y))
        y += title_card.height + SPACING_LARGE
        
        # Left panel: Topic list card
        left_panel_x = 50
        topic_list_card = pygame.Rect(left_panel_x, y, 320, 550)
        Card.draw(self.screen, topic_list_card, COLORS['surface'], COLORS['border'], shadow=True)
        header_rect = Card.draw_header(self.screen, topic_list_card, "Available Topics", self.font_medium)
        
        self.topic_list.rect.x = left_panel_x + PADDING_MEDIUM
        self.topic_list.rect.y = y + header_rect.height + PADDING_SMALL
        self.topic_list.rect.width = topic_list_card.width - PADDING_MEDIUM * 2
        self.topic_list.rect.height = topic_list_card.height - header_rect.height - PADDING_MEDIUM * 2
        self.topic_list.draw(self.screen, self.font_small, self.font_small)
        
        # Right panel: Configuration and editor
        x_right = left_panel_x + topic_list_card.width + SPACING_LARGE
        y_right = y
        right_panel_width = self.screen_width - x_right - 50
        
        # Topic configuration card
        config_card = pygame.Rect(x_right, y_right, right_panel_width, 120)
        Card.draw(self.screen, config_card, COLORS['surface'], COLORS['border'], shadow=True)
        header_rect = Card.draw_header(self.screen, config_card, "Topic Configuration", self.font_medium)
        
        card_y = config_card.y + header_rect.height + SPACING_MEDIUM
        
        # Topic Name input
        label = self.font_medium.render("Topic Name:", True, COLORS['text'])
        self.screen.blit(label, (x_right + PADDING_MEDIUM, card_y))
        self.control_topic_input.rect.x = x_right + PADDING_MEDIUM
        self.control_topic_input.rect.y = card_y + label.get_height() + SPACING_SMALL
        self.control_topic_input.rect.width = (right_panel_width - PADDING_MEDIUM * 3) // 2
        self.control_topic_input.rect.height = 40
        self.control_topic_input.draw(self.screen, self.font_medium)
        
        # Topic Type input
        label2 = self.font_medium.render("Topic Type:", True, COLORS['text'])
        label2_x = x_right + PADDING_MEDIUM + self.control_topic_input.rect.width + SPACING_MEDIUM
        self.screen.blit(label2, (label2_x, card_y))
        self.control_type_input.rect.x = label2_x
        self.control_type_input.rect.y = card_y + label2.get_height() + SPACING_SMALL
        self.control_type_input.rect.width = right_panel_width - label2_x - PADDING_MEDIUM
        self.control_type_input.rect.height = 40
        self.control_type_input.draw(self.screen, self.font_medium)
        
        y_right = config_card.y + config_card.height + SPACING_LARGE
        
        # JSON editor card
        editor_card = pygame.Rect(x_right, y_right, right_panel_width, 380)
        Card.draw(self.screen, editor_card, COLORS['surface'], COLORS['border'], shadow=True)
        header_rect = Card.draw_header(self.screen, editor_card, "Message Content (JSON)", self.font_medium)
        
        self.json_editor.rect.x = x_right + PADDING_MEDIUM
        self.json_editor.rect.y = y_right + header_rect.height + PADDING_SMALL
        self.json_editor.rect.width = right_panel_width - PADDING_MEDIUM * 2
        self.json_editor.rect.height = editor_card.height - header_rect.height - PADDING_MEDIUM * 2
        self.json_editor.draw(self.screen, self.font_small)
        y_right += editor_card.height + SPACING_LARGE
        
        # Action buttons card
        button_card = pygame.Rect(x_right, y_right, right_panel_width, 80)
        Card.draw(self.screen, button_card, COLORS['surface'], COLORS['border'], shadow=True)
        
        button_y = button_card.y + PADDING_MEDIUM
        button_x = x_right + PADDING_MEDIUM
        button_spacing = SPACING_MEDIUM
        
        # Preset buttons (Takeoff, Land, Return, Hover)
        for i, btn in enumerate(self.preset_buttons):
            btn.rect.x = button_x
            btn.rect.y = button_y
            btn.rect.width = 120
            btn.rect.height = 40
            btn.draw(self.screen, self.font_medium)
            button_x += btn.rect.width + button_spacing
        
        # Format and Send buttons (positioned after preset buttons)
        self.format_json_btn.rect.x = button_x
        self.format_json_btn.rect.y = button_y
        self.format_json_btn.rect.width = 140
        self.format_json_btn.rect.height = 40
        self.format_json_btn.draw(self.screen, self.font_medium)
        
        self.send_command_btn.rect.x = button_x + self.format_json_btn.rect.width + button_spacing
        self.send_command_btn.rect.y = button_y
        self.send_command_btn.rect.width = 160
        self.send_command_btn.rect.height = 40
        self.send_command_btn.draw(self.screen, self.font_medium)
        
        # Command history card at bottom
        y = y_right + button_card.height + SPACING_LARGE
        history_card = pygame.Rect(50, y, self.screen_width - 100, self.screen_height - y - SPACING_MEDIUM)
        Card.draw(self.screen, history_card, COLORS['surface'], COLORS['border'], shadow=True)
        header_rect = Card.draw_header(self.screen, history_card, "Command History", self.font_medium)
        
        history_area = pygame.Rect(history_card.x + PADDING_MEDIUM, history_card.y + header_rect.height + PADDING_MEDIUM,
                                   history_card.width - PADDING_MEDIUM * 2, 
                                   history_card.height - header_rect.height - PADDING_MEDIUM * 2)
        pygame.draw.rect(self.screen, COLORS['bg'], history_area, border_radius=8)
        
        history_y = history_area.y + PADDING_SMALL
        for cmd in self.command_history[-15:]:  # Show last 15 commands
            cmd_surface = self.font_small.render(cmd, True, COLORS['text_secondary'])
            if history_y + cmd_surface.get_height() > history_area.bottom - PADDING_SMALL:
                break
            self.screen.blit(cmd_surface, (history_area.x + PADDING_MEDIUM, history_y))
            history_y += cmd_surface.get_height() + SPACING_XS
                
    def draw_pointcloud_tab(self):
        """Draw point cloud display tab with refined card design and proper spacing."""
        y = 70
        
        # Title card
        title_card = pygame.Rect(50, y, 500, 50)
        Card.draw(self.screen, title_card, COLORS['surface'], COLORS['border'], shadow=True)
        title = self.font_large.render("Point Cloud Display", True, COLORS['text'])
        title_y = title_card.y + (title_card.height - title.get_height()) // 2
        self.screen.blit(title, (title_card.x + PADDING_MEDIUM, title_y))
        y += title_card.height + SPACING_LARGE
        
        if not HAS_MATPLOTLIB:
            error_card = pygame.Rect(50, y, self.screen_width - 100, 100)
            Card.draw(self.screen, error_card, COLORS['surface'], COLORS['error'], shadow=True)
            error_text = self.font_medium.render("matplotlib and numpy required for point cloud display", 
                                                True, COLORS['error'])
            text_rect = error_text.get_rect(center=error_card.center)
            self.screen.blit(error_text, text_rect)
            return
            
        # Point cloud display area with refined card style and proper padding
        pc_area = pygame.Rect(50, y, self.screen_width - 100, self.screen_height - y - SPACING_MEDIUM)
        Card.draw(self.screen, pc_area, COLORS['surface'], COLORS['border'], shadow=True)
        
        # Inner area with padding
        pc_inner = pygame.Rect(pc_area.x + PADDING_MEDIUM, pc_area.y + PADDING_MEDIUM,
                              pc_area.width - PADDING_MEDIUM * 2, pc_area.height - PADDING_MEDIUM * 2)
        
        if self.pc_surface:
            # Scale and center point cloud image
            pc_rect = self.pc_surface.get_rect()
            scale = min(pc_inner.width / pc_rect.width, pc_inner.height / pc_rect.height)
            new_size = (int(pc_rect.width * scale), int(pc_rect.height * scale))
            scaled_pc = pygame.transform.scale(self.pc_surface, new_size)
            scaled_rect = scaled_pc.get_rect(center=pc_inner.center)
            self.screen.blit(scaled_pc, scaled_rect)
        else:
            # Placeholder with refined card-style message
            placeholder_card = pygame.Rect(pc_inner.centerx - 250, pc_inner.centery - 50, 500, 100)
            Card.draw(self.screen, placeholder_card, COLORS['surface_light'], COLORS['border'], shadow=False)
            placeholder_text = self.font_medium.render("Waiting for point cloud data...", 
                                                      True, COLORS['text_secondary'])
            placeholder_rect = placeholder_text.get_rect(center=(placeholder_card.centerx, placeholder_card.centery - 10))
            self.screen.blit(placeholder_text, placeholder_rect)
            
            if self.is_connected:
                loading_text = self.font_small.render("Fetching point cloud...", 
                                                      True, COLORS['primary'])
                loading_rect = loading_text.get_rect(center=(placeholder_card.centerx, placeholder_card.centery + 15))
                self.screen.blit(loading_text, loading_rect)
                
    def draw_network_tab(self):
        """Draw network test tab with refined card design and proper spacing."""
        y = 70
        
        # Title card
        title_card = pygame.Rect(50, y, 500, 50)
        Card.draw(self.screen, title_card, COLORS['surface'], COLORS['border'], shadow=True)
        title = self.font_large.render("Network Test", True, COLORS['text'])
        title_y = title_card.y + (title_card.height - title.get_height()) // 2
        self.screen.blit(title, (title_card.x + PADDING_MEDIUM, title_y))
        y += title_card.height + SPACING_LARGE
        
        # Test configuration card
        config_card = pygame.Rect(50, y, self.screen_width - 100, 200)
        Card.draw(self.screen, config_card, COLORS['surface'], COLORS['border'], shadow=True)
        header_rect = Card.draw_header(self.screen, config_card, "Test Configuration", self.font_medium)
        
        card_y = config_card.y + header_rect.height + SPACING_MEDIUM
        
        # Test URL
        label = self.font_medium.render("Test Address:", True, COLORS['text'])
        self.screen.blit(label, (config_card.x + PADDING_MEDIUM, card_y))
        self.test_url_input.rect.x = config_card.x + PADDING_MEDIUM
        self.test_url_input.rect.y = card_y + label.get_height() + SPACING_SMALL
        self.test_url_input.rect.width = config_card.width - PADDING_MEDIUM * 2
        self.test_url_input.rect.height = 35
        self.test_url_input.draw(self.screen, self.font_medium)
        card_y += label.get_height() + SPACING_SMALL + self.test_url_input.rect.height + SPACING_MEDIUM
        
        # Timeout and Test button row
        label2 = self.font_medium.render("Timeout (seconds):", True, COLORS['text'])
        self.screen.blit(label2, (config_card.x + PADDING_MEDIUM, card_y))
        self.test_timeout_input.rect.x = config_card.x + PADDING_MEDIUM
        self.test_timeout_input.rect.y = card_y + label2.get_height() + SPACING_SMALL
        self.test_timeout_input.rect.width = 200
        self.test_timeout_input.rect.height = 35
        self.test_timeout_input.draw(self.screen, self.font_medium)
        
        # Test button
        self.test_btn.rect.x = config_card.x + PADDING_MEDIUM + self.test_timeout_input.rect.width + SPACING_MEDIUM
        self.test_btn.rect.y = card_y + label2.get_height() + SPACING_SMALL
        self.test_btn.rect.width = 150
        self.test_btn.rect.height = 35
        self.test_btn.draw(self.screen, self.font_medium)
        
        y += config_card.height + SPACING_LARGE
        
        # Test results card
        result_card = pygame.Rect(50, y, self.screen_width - 100, self.screen_height - y - SPACING_MEDIUM)
        Card.draw(self.screen, result_card, COLORS['surface'], COLORS['border'], shadow=True)
        header_rect = Card.draw_header(self.screen, result_card, "Test Results", self.font_medium)
        
        result_area = pygame.Rect(result_card.x + PADDING_MEDIUM, result_card.y + header_rect.height + PADDING_MEDIUM,
                                 result_card.width - PADDING_MEDIUM * 2, 
                                 result_card.height - header_rect.height - PADDING_MEDIUM * 2)
        pygame.draw.rect(self.screen, COLORS['bg'], result_area, border_radius=8)
        
        result_y = result_area.y + PADDING_SMALL
        for result in self.test_results[-25:]:  # Show last 25 results
            result_surface = self.font_small.render(result, True, COLORS['text_secondary'])
            if result_y + result_surface.get_height() > result_area.bottom - PADDING_SMALL:
                break
            self.screen.blit(result_surface, (result_area.x + PADDING_MEDIUM, result_y))
            result_y += result_surface.get_height() + SPACING_XS
                
    def handle_events(self):
        """Handle all pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            # Tab switching
            tab_changed = self.tab_bar.handle_event(event)
            if tab_changed is not None:
                self.current_tab = tab_changed
                
            # Handle tab-specific events
            if self.current_tab == 0:  # Connection tab
                self.connection_url_input.handle_event(event)
                self.use_mock_checkbox.handle_event(event)
                self.connect_btn.handle_event(event)
                self.disconnect_btn.handle_event(event)
            elif self.current_tab == 3:  # Control tab
                selected_topic = self.topic_list.handle_event(event)
                if selected_topic:
                    # selected_topic is a tuple (name, type)
                    if isinstance(selected_topic, tuple):
                        self.control_topic_input.text = selected_topic[0]
                        if selected_topic[1]:  # If type is available
                            self.control_type_input.text = selected_topic[1]
                    else:
                        self.control_topic_input.text = selected_topic
                self.control_topic_input.handle_event(event)
                self.control_type_input.handle_event(event)
                self.json_editor.handle_event(event)
                for btn in self.preset_buttons:
                    btn.handle_event(event)
                self.format_json_btn.handle_event(event)
                self.send_command_btn.handle_event(event)
            elif self.current_tab == 5:  # Network tab
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
        elif self.current_tab == 5:
            self.test_url_input.update(self.dt)
            self.test_timeout_input.update(self.dt)
            self.test_btn.update(self.dt)
            
    def draw(self):
        """Draw everything with refined design."""
        # Clear screen with subtle gradient effect
        self.screen.fill(COLORS['bg'])
        
        # Draw subtle background pattern for depth
        for y in range(0, self.screen_height, 60):
            pygame.draw.line(self.screen, COLORS['bg_secondary'], (0, y), (self.screen_width, y), 1)
        
        # Draw tab bar
        self.tab_bar.draw(self.screen, self.font_medium)
        
        # Draw tab content with subtle border
        content_area = pygame.Rect(0, 50, self.screen_width, self.screen_height - 50)
        pygame.draw.rect(self.screen, COLORS['bg'], content_area)
        
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
        pygame.quit()


def main():
    """Main entry point."""
    app = RosClientPygameGUI()
    app.run()


if __name__ == "__main__":
    main()

