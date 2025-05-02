import pygame
import random
import time

class DisplayManager:
    def __init__(self):
        self.font = pygame.font.Font(None, 29)
        self.player_positions = ["Top Left", "Top Mid", "Top Right", "Center", "Center Down"]
        self.reward_positions = ["Left", "Right", "Unknown"]
    
    def _get_text_surface(self, text, color=(0, 0, 0)):
        """Get a cached text surface or create a new one"""
        return self.font.render(text, True, color)
    
    def _draw_table_header(self, screen, title, y_offset):
        """Draw a table header and return the new y_offset"""
        text_surface = self._get_text_surface(title)
        screen.blit(text_surface, (10, y_offset))
        y_offset += 30
        
        # Draw column headers
        for col, reward in enumerate(self.reward_positions):
            text_surface = self._get_text_surface(reward)
            screen.blit(text_surface, (200 + col * 100, y_offset))
        
        return y_offset + 30
    
    def _draw_table_row(self, screen, row_text, values, y_offset, colors=None):
        """Draw a table row and return the new y_offset"""
        # Draw row header
        text_surface = self._get_text_surface(row_text)
        screen.blit(text_surface, (10, y_offset))
        
        # Draw values
        for col, value in enumerate(values):
            color = colors[col] if colors else (0, 0, 0)
            text = f"{value:.2f}"
            text_surface = self._get_text_surface(text, color)
            screen.blit(text_surface, (200 + col * 100, y_offset))
        
        return y_offset + 30

    def display_qx_text(self, world, agent, qx_differences):
        """Display the Q(x) distribution"""
        screen = world._environment.get_display()
        
        # Get current observation
        y = world.observe()
        
        # Calculate probability vectors
        qx_vector = [float(round(agent.qx.probability(x), 2)) for x in range(15)]
        px_vector = [float(round(agent.px.probability(x), 2)) for x in range(15)]
        py_x_vector = [float(round(agent.py_x(x).probability(y), 2)) for x in range(15)]
        
        # Draw Q(x) table
        y_offset = 10
        y_offset = self._draw_table_header(screen, "Q(x) Distribution", y_offset)
        
        # Draw Q(x) rows with color changes
        for row, player in enumerate(self.player_positions):
            row_values = []
            row_colors = []
            for col in range(3):
                state_idx = row * 3 + col
                row_values.append(qx_vector[state_idx])
                # Determine color based on difference
                diff = qx_differences[state_idx]
                if diff > 0:
                    color = (0, 255, 0)  # Green for increase
                elif diff < 0:
                    color = (255, 0, 0)  # Red for decrease
                else:
                    color = (0, 0, 0)  # Black for no change
                row_colors.append(color)
            
            y_offset = self._draw_table_row(screen, player, row_values, y_offset, colors=row_colors)
        
        # Draw P(x) table
        y_offset += 15
        y_offset = self._draw_table_header(screen, "P(x) Distribution", y_offset)
        
        # Draw P(x) rows
        for row, player in enumerate(self.player_positions):
            row_values = []
            for col in range(3):
                state_idx = row * 3 + col
                row_values.append(px_vector[state_idx])
            y_offset = self._draw_table_row(screen, player, row_values, y_offset)
        
        # Draw P(y|x) table
        y_offset += 15
        y_offset = self._draw_table_header(screen, f"P(y={y}|x) Distribution", y_offset)
        
        # Draw P(y|x) rows
        for row, player in enumerate(self.player_positions):
            row_values = []
            for col in range(3):
                state_idx = row * 3 + col
                row_values.append(py_x_vector[state_idx])
            y_offset = self._draw_table_row(screen, player, row_values, y_offset)

# Create a singleton instance
_display_manager = None

def display_qx_text(world, agent, qx_differences):
    """Wrapper function to use the display manager"""
    global _display_manager
    if _display_manager is None:
        _display_manager = DisplayManager()
    _display_manager.display_qx_text(world, agent, qx_differences) 