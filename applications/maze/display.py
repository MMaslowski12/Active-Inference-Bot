import pygame
import random
import time
import numpy as np
from core.distributions import DiscreteDistribution
from applications.maze.generative_model.mapping import state_to_index, index_to_state, get_observation_idx, NUM_PLAYER_OBS, NUM_STIMULI
eps = 1e-16

class DisplayManager:
    def __init__(self):
        self.font = pygame.font.Font(None, 29)
        self.player_positions = ["Top Left", "Top Mid", "Top Right", "Center", "Center Down"]
        self.reward_positions = ["Left", "Right"]
        self.stimulus_types = ["Positive", "Neutral", "Negative"]
        self.WINDOW_WIDTH = 1300
        self.current_policy = 0  # 0: up, 1: down, 2: left, 3: right
        self.display_mode = "standard"  # Current display mode for right panel
        
        # Sequence of moves for alternative mode
        self.move_sequence = [0, 0, 0, 0]  # Initial sequence: all "Up"
        self.action_names = ["Up", "Down", "Left", "Right"]
        
        # Define button dimensions and positions
        self.button_width = 80
        self.button_height = 30
        self.button_margin = 10
        self.button_y = 10
        
        # Calculate button positions - right-aligned
        total_width = 5 * self.button_width + 4 * self.button_margin  # Back to 5 buttons
        start_x = self.WINDOW_WIDTH - total_width - 10  # 10px margin from right edge
        
        self.button_x_positions = [
            start_x,  # First button
            start_x + self.button_width + self.button_margin,  # Second button
            start_x + 2 * (self.button_width + self.button_margin),  # Third button
            start_x + 3 * (self.button_width + self.button_margin),   # Fourth button
            start_x + 4 * (self.button_width + self.button_margin)    # Mode switch button
        ]
    
    def _get_text_surface(self, text, color=(0, 0, 0)):
        """Get a cached text surface or create a new one"""
        return self.font.render(text, True, color)
    
    def _draw_button(self, screen, text, x, y, width, height, is_selected=False):
        """Draw a button and return its rect"""
        # Draw button background
        color = (200, 200, 200) if is_selected else (150, 150, 150)
        pygame.draw.rect(screen, color, (x, y, width, height))
        pygame.draw.rect(screen, (0, 0, 0), (x, y, width, height), 2)  # Border
        
        # Draw button text
        text_surface = self._get_text_surface(text)
        text_rect = text_surface.get_rect(center=(x + width/2, y + height/2))
        screen.blit(text_surface, text_rect)
        
        return pygame.Rect(x, y, width, height)
    
    def _draw_table_header(self, screen, title, y_offset, is_observation=False):
        """Draw a table header and return the new y_offset"""
        text_surface = self._get_text_surface(title)
        screen.blit(text_surface, (10, y_offset))
        y_offset += 30
        
        # Draw column headers
        if is_observation:
            # For observation tables, show stimulus types
            for col, stimulus in enumerate(self.stimulus_types):
                text_surface = self._get_text_surface(stimulus)
                screen.blit(text_surface, (200 + col * 100, y_offset))
        else:
            # For state tables, show reward positions
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

    def handle_click(self, pos):
        """Handle mouse click on policy buttons"""
        x, y = pos
        
        for i, button_x in enumerate(self.button_x_positions):
            button_rect = pygame.Rect(button_x, self.button_y, self.button_width, self.button_height)
            if button_rect.collidepoint(x, y):
                if i < 4:  # Policy buttons
                    if self.display_mode == "standard":
                        old_policy = self.current_policy
                        self.current_policy = i
                    else:
                        # In alternative mode, cycle through moves
                        self.move_sequence[i] = (self.move_sequence[i] + 1) % 4
                elif i == 4:  # Mode switch button
                    self.display_mode = "alternative" if self.display_mode == "standard" else "standard"
                return True
        return False

    def _draw_standard_policy_buttons(self, screen, action_names):
        """Draw the standard policy buttons"""
        for i, name in enumerate(action_names):
            button_rect = self._draw_button(screen, name, 
                            self.button_x_positions[i], 
                            self.button_y,
                            self.button_width, 
                            self.button_height,
                            is_selected=(i == self.current_policy))
            
            # Draw button border for debugging
            pygame.draw.rect(screen, (255, 0, 0), button_rect, 1)

    def _draw_alternative_policy_buttons(self, screen, action_names):
        """Draw the alternative policy buttons showing move sequence"""
        # Draw sequence number labels
        small_font = pygame.font.Font(None, 20)
        for i in range(4):
            # Draw sequence number above button
            text = f"Move {i+1}"
            text_surface = small_font.render(text, True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=(self.button_x_positions[i] + self.button_width/2, 
                                                     self.button_y - 15))
            screen.blit(text_surface, text_rect)
            
            # Draw the button with current move
            current_move = self.move_sequence[i]
            button_rect = self._draw_button(screen, 
                            self.action_names[current_move],
                            self.button_x_positions[i], 
                            self.button_y,
                            self.button_width, 
                            self.button_height,
                            is_selected=True)  # Always selected to show it's active
            
            # Draw button border
            pygame.draw.rect(screen, (0, 255, 0), button_rect, 1)

    def _draw_standard_tables(self, screen, efe_values, s_pi_t_values, entropy_values, o_pi_t_values, zeta_values, action_names):
        """Draw the standard tables layout"""
        # Use a smaller font for the right panel
        small_font = pygame.font.Font(None, 22)
        def get_small_text_surface(text, color=(0, 0, 0)):
            return small_font.render(text, True, color)
        
        # Draw EFE table below buttons (right-aligned)
        efe_x_offset = self.button_x_positions[0]
        label_col_width = 100
        col_width = 60
        row_height = 22
        y_offset = self.button_y + self.button_height + 10
        text_surface = get_small_text_surface(f"Expected Free Energy (EFE) - Policy: {action_names[self.current_policy]}")
        screen.blit(text_surface, (efe_x_offset, y_offset))
        y_offset += row_height
        
        # Draw current policy EFE value
        text = f"{efe_values[0]:.2f}"
        text_surface = get_small_text_surface(text)
        screen.blit(text_surface, (efe_x_offset + label_col_width + col_width * 2, y_offset))
        y_offset += row_height
        
        # Helper to draw a 2x5 table for state-based values (s_pi_t and entropy)
        def draw_state_table(title, values, y_offset):
            # Draw header
            text_surface = get_small_text_surface(title)
            screen.blit(text_surface, (efe_x_offset, y_offset))
            y_offset += row_height
            
            # Draw column headers for reward positions
            for col, reward in enumerate(self.reward_positions):
                text_surface = get_small_text_surface(reward)
                screen.blit(text_surface, (efe_x_offset + label_col_width + col * col_width, y_offset))
            y_offset += row_height
            
            # Draw rows for each player position
            for row, player in enumerate(self.player_positions):
                text_surface = get_small_text_surface(player)
                screen.blit(text_surface, (efe_x_offset, y_offset))
                
                # Get values for this row using state mapping
                row_values = []
                for col in range(2):
                    state_idx = state_to_index(row, col)
                    row_values.append(values[state_idx])
                
                # Draw values
                for col, value in enumerate(row_values):
                    text = f"{value:.2f}"
                    text_surface = get_small_text_surface(text)
                    screen.blit(text_surface, (efe_x_offset + label_col_width + col * col_width, y_offset))
                y_offset += row_height
            
            return y_offset + 5

        # Helper to draw a 3x6 table for observation-based values (o_pi_t and zeta)
        def draw_observation_table(title, values, y_offset):
            # Draw header
            text_surface = get_small_text_surface(title)
            screen.blit(text_surface, (efe_x_offset, y_offset))
            y_offset += row_height
            
            # Draw column headers for stimulus types
            for col, stimulus in enumerate(self.stimulus_types):
                text_surface = get_small_text_surface(stimulus)
                screen.blit(text_surface, (efe_x_offset + label_col_width + col * col_width, y_offset))
            y_offset += row_height
            
            # Draw rows for each player position plus CD (Right)
            for row, player in enumerate(self.player_positions + ["CD (Right)"]):
                text_surface = get_small_text_surface(player)
                screen.blit(text_surface, (efe_x_offset, y_offset))
                
                # Get values for this row using observation mapping
                row_values = []
                for col in range(NUM_STIMULI):
                    obs_idx = get_observation_idx(row, col)
                    row_values.append(values[obs_idx])
                
                # Draw values
                for col, value in enumerate(row_values):
                    text = f"{value:.2f}"
                    text_surface = get_small_text_surface(text)
                    screen.blit(text_surface, (efe_x_offset + label_col_width + col * col_width, y_offset))
                y_offset += row_height
            
            return y_offset + 5
        
        # Draw s_pi_t, entropy, o_pi_t, zeta as tables (right-aligned, compact)
        y_offset = draw_state_table("s_pi_t", s_pi_t_values[0], y_offset)
        y_offset = draw_state_table("entropy", entropy_values[0], y_offset)
        y_offset = draw_observation_table("o_pi_t", o_pi_t_values[0], y_offset)
        y_offset = draw_observation_table("zeta", zeta_values[0], y_offset)

    def _draw_alternative_tables(self, screen, efe_values, s_pi_t_values, entropy_values, o_pi_t_values, zeta_values, action_names):
        """Draw the alternative tables layout"""
        # Use a smaller font for the right panel
        small_font = pygame.font.Font(None, 22)
        def get_small_text_surface(text, color=(0, 0, 0)):
            return small_font.render(text, True, color)
        
        # Draw EFE table below buttons (right-aligned)
        efe_x_offset = self.button_x_positions[0]
        label_col_width = 100
        col_width = 60
        row_height = 22
        y_offset = self.button_y + self.button_height + 10
        
        # Draw header with alternative style
        move_sequence_str = " -> ".join([self.action_names[move] for move in self.move_sequence])
        text_surface = get_small_text_surface(f"Alternative EFE - Sequence: {move_sequence_str}")
        screen.blit(text_surface, (efe_x_offset, y_offset))
        y_offset += row_height
        
        # Draw EFE value
        text = f"{efe_values[0]:.2f}"
        text_surface = get_small_text_surface(text)
        screen.blit(text_surface, (efe_x_offset + label_col_width + col_width * 2, y_offset))
        y_offset += row_height
        
        # Helper to draw a 2x5 table for state-based values (s_pi_t and entropy)
        def draw_state_table(title, values, y_offset):
            # Draw header
            text_surface = get_small_text_surface(title)
            screen.blit(text_surface, (efe_x_offset, y_offset))
            y_offset += row_height
            
            # Draw column headers for reward positions
            for col, reward in enumerate(self.reward_positions):
                text_surface = get_small_text_surface(reward)
                screen.blit(text_surface, (efe_x_offset + label_col_width + col * col_width, y_offset))
            y_offset += row_height
            
            # Draw rows for each player position with actual values
            for row, player in enumerate(self.player_positions):
                text_surface = get_small_text_surface(player)
                screen.blit(text_surface, (efe_x_offset, y_offset))
                
                # Get values for this row using state mapping
                row_values = []
                for col in range(2):
                    state_idx = state_to_index(row, col)
                    row_values.append(values[state_idx])
                
                # Draw values
                for col, value in enumerate(row_values):
                    text = f"{value:.2f}"
                    text_surface = get_small_text_surface(text)
                    screen.blit(text_surface, (efe_x_offset + label_col_width + col * col_width, y_offset))
                y_offset += row_height
            
            return y_offset + 5

        # Helper to draw a 3x6 table for observation-based values (o_pi_t and zeta)
        def draw_observation_table(title, values, y_offset):
            # Draw header
            text_surface = get_small_text_surface(title)
            screen.blit(text_surface, (efe_x_offset, y_offset))
            y_offset += row_height
            
            # Draw column headers for stimulus types
            for col, stimulus in enumerate(self.stimulus_types):
                text_surface = get_small_text_surface(stimulus)
                screen.blit(text_surface, (efe_x_offset + label_col_width + col * col_width, y_offset))
            y_offset += row_height
            
            # Draw rows for each player position plus CD (Right) with actual values
            for row, player in enumerate(self.player_positions + ["CD (Right)"]):
                text_surface = get_small_text_surface(player)
                screen.blit(text_surface, (efe_x_offset, y_offset))
                
                # Get values for this row using observation mapping
                row_values = []
                for col in range(NUM_STIMULI):
                    obs_idx = get_observation_idx(row, col)
                    row_values.append(values[obs_idx])
                
                # Draw values
                for col, value in enumerate(row_values):
                    text = f"{value:.2f}"
                    text_surface = get_small_text_surface(text)
                    screen.blit(text_surface, (efe_x_offset + label_col_width + col * col_width, y_offset))
                y_offset += row_height
            
            return y_offset + 5
        
        # Draw s_pi_t, entropy, o_pi_t, zeta as tables (right-aligned, compact)
        y_offset = draw_state_table("s_pi_t (Alt)", s_pi_t_values[0], y_offset)
        y_offset = draw_state_table("entropy (Alt)", entropy_values[0], y_offset)
        y_offset = draw_observation_table("o_pi_t (Alt)", o_pi_t_values[0], y_offset)
        y_offset = draw_observation_table("zeta (Alt)", zeta_values[0], y_offset)

    def display_qx_text(self, world, agent, qx_differences):
        """Display the Q(x) distribution"""
        screen = world._environment.get_display()
        
        # Get current observation
        y = world.observe()
        
        # Get current state from agent's q(x) distribution
        current_state = np.argmax(agent.qx.get_probabilities())
        
        # Calculate probability vectors
        qx_vector = [float(round(agent.qx.probability(x), 2)) for x in range(10)]
        px_vector = [float(round(agent.px.probability(x), 2)) for x in range(10)]
        py_x_vector = [float(round(agent.py_x(x).probability(y), 2)) for x in range(10)]
        
        # Calculate EFE for each action
        efe_values = []
        s_pi_t_values = []
        entropy_values = []
        o_pi_t_values = []
        zeta_values = []
        
        # Calculate components for current policy only
        if self.display_mode == "standard":
            # Standard mode: single action
            action = np.zeros(4)
            action[self.current_policy] = 1
            pi = lambda tau: action
            tau = 1  # Use tau=1 for standard mode
        else:
            # Alternative mode: sequence of actions
            def sequence_pi(tau):
                if tau >= len(self.move_sequence):
                    return np.zeros(4)  # Return no action if beyond sequence
                action = np.zeros(4)
                action[self.move_sequence[tau]] = 1
                return action
            pi = sequence_pi
            tau = 4  # Use tau=4 for alternative mode
        
        # Calculate components using agent's q(x) directly
        s_pi_t = agent._get_s_pi_t(state=agent.qx, pi=pi, tau=tau)
        o_pi_t = agent._get_o_pi_t(s_pi_t)
        entropy = agent.calculate_entropy()
        c_t = agent.c.get_probabilities()
        zeta = np.log(o_pi_t+eps)-np.log(c_t+eps)
        
        # Store values
        efe = agent.calculate_efe(state=agent.qx, pi=pi, tau=tau)
        efe_values.append(float(round(efe, 2)))
        s_pi_t_values.append(s_pi_t)
        entropy_values.append(entropy)
        o_pi_t_values.append(o_pi_t)
        zeta_values.append(zeta)
        
        # Clear the right side of the screen
        pygame.draw.rect(screen, (200, 200, 200), (self.button_x_positions[0] - 10, 0, 
                                                  self.WINDOW_WIDTH - self.button_x_positions[0] + 20, 
                                                  self.WINDOW_WIDTH))
        
        # Draw mode switch button
        mode_text = "Alt" if self.display_mode == "standard" else "Std"
        self._draw_button(screen, mode_text,
                        self.button_x_positions[4],
                        self.button_y,
                        self.button_width,
                        self.button_height,
                        is_selected=self.display_mode == "alternative")
        
        # Show appropriate panel based on mode
        action_names = ["Up", "Down", "Left", "Right"]
        
        if self.display_mode == "standard":
            # Draw standard policy buttons and tables
            self._draw_standard_policy_buttons(screen, action_names)
            self._draw_standard_tables(screen, efe_values, s_pi_t_values, entropy_values, o_pi_t_values, zeta_values, action_names)
        else:
            # Draw alternative policy buttons and tables
            self._draw_alternative_policy_buttons(screen, action_names)
            self._draw_alternative_tables(screen, efe_values, s_pi_t_values, entropy_values, o_pi_t_values, zeta_values, action_names)
        
        # Reset y_offset for other tables
        y_offset = 10
        
        # Draw Q(x) table
        y_offset = self._draw_table_header(screen, "Q(x) Distribution", y_offset)
        for position, player in enumerate(self.player_positions):
            row_values = []
            row_colors = []
            for reward in range(2):
                state_idx = state_to_index(position, reward)
                row_values.append(qx_vector[state_idx])
                diff = qx_differences[state_idx]
                if diff > 0:
                    color = (0, 255, 0)
                elif diff < 0:
                    color = (255, 0, 0)
                else:
                    color = (0, 0, 0)
                row_colors.append(color)
            y_offset = self._draw_table_row(screen, player, row_values, y_offset, colors=row_colors)
        
        # Draw P(x) table
        y_offset += 10
        y_offset = self._draw_table_header(screen, "P(x) Distribution", y_offset)
        for position, player in enumerate(self.player_positions):
            row_values = []
            for reward in range(2):
                state_idx = state_to_index(position, reward)
                row_values.append(px_vector[state_idx])
            y_offset = self._draw_table_row(screen, player, row_values, y_offset)
        
        # Draw P(y|x) table
        y_offset += 10
        y_offset = self._draw_table_header(screen, f"P(y={y}|x) Distribution", y_offset)
        for position, player in enumerate(self.player_positions):
            row_values = []
            for reward in range(2):
                state_idx = state_to_index(position, reward)
                row_values.append(py_x_vector[state_idx])
            y_offset = self._draw_table_row(screen, player, row_values, y_offset)

# Create a singleton instance
_display_manager = None

def get_display_manager():
    """Get the singleton display manager instance"""
    global _display_manager
    if _display_manager is None:
        _display_manager = DisplayManager()
    return _display_manager

def display_qx_text(world, agent, qx_differences):
    """Wrapper function to use the display manager"""
    display_manager = get_display_manager()
    display_manager.display_qx_text(world, agent, qx_differences) 