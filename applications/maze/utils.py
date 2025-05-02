import pygame

def handle_input():
    """Handle keyboard input for player movement."""
    keys = pygame.key.get_pressed()
    
    # Check each direction separately and only move in one direction
    if keys[pygame.K_UP]:
        return (0, -1)
    elif keys[pygame.K_DOWN]:
        return (0, 1)
    elif keys[pygame.K_LEFT]:
        return (-1, 0)
    elif keys[pygame.K_RIGHT]:
        return (1, 0)
    
    # Return (0, 0) if no movement keys are pressed
    return (0, 0)