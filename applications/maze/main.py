from applications.maze.environment import MazeGame
from applications.maze.world import MazeWorld
from applications.maze.generative_model.matrices import observation_matrix, priors_vector, c_vector
from applications.maze.generative_model.transitioner import transitioner
from applications.maze.display import display_qx_text, get_display_manager
from agents.discrete_agent import DiscreteAgent
from applications.maze.utils import handle_input
import pygame
import numpy as np

def run_maze_game():
    world = MazeWorld(environment=MazeGame(), machina_type='matrix', A=observation_matrix)
    agent = DiscreteAgent(px_vector=priors_vector, c_vector=c_vector, transitioner=transitioner, machina_type='matrix', A=observation_matrix, q_learning_rate=10)
    clock = pygame.time.Clock()
    
    # Get display manager
    display_manager = get_display_manager()
    
    while True:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    display_manager.handle_click(event.pos)
        
        # Handle keyboard input
        action = handle_input()
        world.step(action)
        y = world.observe()
        
        # Store previous Qx values
        prev_qx = np.round(agent.qx.get_probabilities(), 3)
        
        for _ in range(1): #20
            agent.adjust_q(y)
            
        # Calculate differences in Q(x) distribution
        qx_differences = np.abs(np.round(agent.qx.get_probabilities(), 3) - prev_qx)
        prev_qx = agent.qx.get_probabilities()
        
        # Draw everything before flipping
        world.display()  # This no longer flips the display
        display_qx_text(world, agent, qx_differences)  # This updates only changed regions
        
        # Single synchronized display update
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(300)

if __name__ == "__main__":
    run_maze_game() 