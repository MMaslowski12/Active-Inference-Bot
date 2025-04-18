from applications.maze.environment import MazeGame
import pygame

def handle_input(game):
    """Handle keyboard input for player movement."""
    keys = pygame.key.get_pressed()
    
    # Check each direction separately and only move in one direction
    if keys[pygame.K_UP]:
        game.move_player(0, -1)
    elif keys[pygame.K_DOWN]:
        game.move_player(0, 1)
    elif keys[pygame.K_LEFT]:
        game.move_player(-1, 0)
    elif keys[pygame.K_RIGHT]:
        game.move_player(1, 0)

def run_maze_game():
    game = MazeGame()
    clock = pygame.time.Clock()
    
    while True:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # Handle keyboard input
        handle_input(game)
        
        # Update display
        game.display()
        
        # Cap the frame rate
        clock.tick(60)

if __name__ == "__main__":
    run_maze_game() 