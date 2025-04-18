import pygame
import sys
import random

class MazeGame:
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        
        # Constants
        self.TILE_SIZE = 60  # Increased tile size for better visibility
        self.WINDOW_WIDTH = 800
        self.WINDOW_HEIGHT = 600
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (200, 200, 200)  # Background color
        self.DARK_GRAY = (100, 100, 100)  # Border color
        self.TILE_COLOR = (240, 240, 240)  # Light gray for tiles
        self.YELLOW = (255, 255, 0)
        self.GREEN = (0, 255, 0)
        
        # Set up the display
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("T-Shaped Maze")
        
        # Calculate maze dimensions
        self.maze_width = 3 * self.TILE_SIZE
        self.maze_height = 3 * self.TILE_SIZE
        
        # Calculate maze position (centered)
        self.maze_x = (self.WINDOW_WIDTH - self.maze_width) // 2
        self.maze_y = (self.WINDOW_HEIGHT - self.maze_height) // 2
        
        # Player properties
        self.player_radius = 12
        # Track player position in grid coordinates (column, row)
        self.current_col = 1  # Middle column
        self.current_row = 1  # One tile up from bottom
        self.update_player_pixel_position()
        
        # Question mark position (bottom tile)
        self.question_x = self.maze_x + self.TILE_SIZE * 1.5
        self.question_y = self.maze_y + self.TILE_SIZE * 2.5  # Adjusted for shorter maze
        
        # Randomly place snack in either top-left or top-right corner
        self.snack_col = random.choice([0, 2])  # 0 for left, 2 for right
        self.snack_row = 0
        self.update_snack_position()
        
        # Movement cooldown
        self.last_move_time = 0
        self.move_cooldown = 150  # milliseconds between moves
        
        # Game state
        self.snack_visible = False
        self.question_mark_visible = True
        
        # Load font for question mark
        self.font = pygame.font.Font(None, 36)
        
    def update_player_pixel_position(self):
        """Update pixel position based on grid position."""
        # For vertical part (including bottom tile)
        if self.current_row >= 1:
            self.player_x = self.maze_x + self.TILE_SIZE * 1.5
            self.player_y = self.maze_y + self.TILE_SIZE * (self.current_row + 0.5)
        # For horizontal part
        else:
            self.player_x = self.maze_x + self.TILE_SIZE * (self.current_col + 0.5)
            self.player_y = self.maze_y + self.TILE_SIZE * 0.5
        
    def update_snack_position(self):
        """Update pixel position of snack based on grid position."""
        self.snack_x = self.maze_x + self.TILE_SIZE * (self.snack_col + 0.5)
        self.snack_y = self.maze_y + self.TILE_SIZE * 0.5
        
    def draw_tile(self, x, y):
        # Draw tile background only
        tile_rect = pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.TILE_COLOR, tile_rect)
        
    def draw_maze(self):
        # Draw all tile backgrounds first
        # Horizontal part of T (top row)
        for i in range(3):
            self.draw_tile(self.maze_x + i * self.TILE_SIZE, self.maze_y)
        
        # Vertical part of T (middle column)
        for i in range(1, 2):  # Adjusted range for shorter maze
            self.draw_tile(self.maze_x + self.TILE_SIZE, self.maze_y + i * self.TILE_SIZE)
        
        # Bottom tile
        self.draw_tile(self.maze_x + self.TILE_SIZE, 
                      self.maze_y + self.TILE_SIZE * 2)  # Adjusted for shorter maze
        
        # Draw grid lines for the T shape
        # Top horizontal part
        pygame.draw.line(self.screen, self.DARK_GRAY, 
                        (self.maze_x, self.maze_y), 
                        (self.maze_x + self.maze_width, self.maze_y))
        pygame.draw.line(self.screen, self.DARK_GRAY, 
                        (self.maze_x, self.maze_y + self.TILE_SIZE), 
                        (self.maze_x + self.maze_width, self.maze_y + self.TILE_SIZE))
        
        # Vertical lines for top part
        for i in range(4):
            x = self.maze_x + i * self.TILE_SIZE
            pygame.draw.line(self.screen, self.DARK_GRAY,
                           (x, self.maze_y),
                           (x, self.maze_y + self.TILE_SIZE))
        
        # Middle and bottom vertical part
        x = self.maze_x + self.TILE_SIZE
        pygame.draw.line(self.screen, self.DARK_GRAY,
                        (x, self.maze_y + self.TILE_SIZE),
                        (x, self.maze_y + self.TILE_SIZE * 3))  # Adjusted for shorter maze
        pygame.draw.line(self.screen, self.DARK_GRAY,
                        (x + self.TILE_SIZE, self.maze_y + self.TILE_SIZE),
                        (x + self.TILE_SIZE, self.maze_y + self.TILE_SIZE * 3))  # Adjusted for shorter maze
        
        # Horizontal lines for vertical part
        for i in range(2, 4):  # Adjusted range for shorter maze
            y = self.maze_y + i * self.TILE_SIZE
            pygame.draw.line(self.screen, self.DARK_GRAY,
                           (x, y),
                           (x + self.TILE_SIZE, y))
        
    def draw_question_mark(self):
        """Draw yellow circle with black question mark if visible."""
        if self.question_mark_visible:
            # Draw yellow circle
            pygame.draw.circle(self.screen, self.YELLOW, 
                             (int(self.question_x), int(self.question_y)), 
                             self.player_radius)
            # Draw question mark
            text = self.font.render("?", True, self.BLACK)
            text_rect = text.get_rect(center=(int(self.question_x), int(self.question_y)))
            self.screen.blit(text, text_rect)
        
    def draw_snack(self):
        """Draw green circle for snack if visible."""
        if self.snack_visible:
            pygame.draw.circle(self.screen, self.GREEN,
                             (int(self.snack_x), int(self.snack_y)),
                             self.player_radius)

    def draw_player(self):
        # Draw player with a slight shadow effect
        pygame.draw.circle(self.screen, (0, 0, 200), 
                         (int(self.player_x), int(self.player_y)), 
                         self.player_radius + 2)  # Shadow
        pygame.draw.circle(self.screen, self.BLUE, 
                         (int(self.player_x), int(self.player_y)), 
                         self.player_radius)  # Player
    
    def check_question_mark_interaction(self):
        """Check if player is on question mark tile and handle interaction."""
        if (self.question_mark_visible and 
            self.current_row == 2 and  # Adjusted for shorter maze
            self.current_col == 1):
            self.question_mark_visible = False
            self.snack_visible = True
            
    def move_player(self, dx, dy):
        """Move player by the given delta in grid coordinates."""
        current_time = pygame.time.get_ticks()
        if current_time - self.last_move_time < self.move_cooldown:
            return
            
        new_col = self.current_col + dx
        new_row = self.current_row + dy
        
        # Check if the move is valid
        valid_move = False
        
        # Moving in vertical part (including bottom tile)
        if self.current_row >= 1:
            if dx == 0:  # Trying to move vertically
                if 1 <= new_row <= 2:  # Adjusted range for shorter maze
                    valid_move = True
                elif new_row == 0 and self.current_row == 1 and self.current_col == 1:
                    # Allow moving up to horizontal part from row 1 in middle column
                    valid_move = True
        # Moving in horizontal part
        elif self.current_row == 0:
            if dy == 0 and 0 <= new_col <= 2:  # Can move left/right in top row
                valid_move = True
            elif dy == 1 and new_col == 1:  # Can move down from middle column
                valid_move = True
            
        if valid_move:
            self.current_col = new_col
            self.current_row = new_row
            self.update_player_pixel_position()
            self.last_move_time = current_time
            self.check_question_mark_interaction()
            
    def get_display(self):
        """Return the pygame display surface."""
        return self.screen
        
    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    
    def display(self):
        # Fill background with gray
        self.screen.fill(self.GRAY)
        
        # Draw maze and game elements
        self.draw_maze()
        self.draw_question_mark()
        self.draw_snack()
        self.draw_player()
        
        # Update display
        pygame.display.flip()
    
    def run(self):
        clock = pygame.time.Clock()
        while True:
            self.update()
            self.display()
            clock.tick(60)  # 60 FPS

if __name__ == "__main__":
    game = MazeGame()
    game.run() 