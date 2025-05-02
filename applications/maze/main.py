from applications.maze.environment import MazeGame
from applications.maze.world import MazeWorld
from applications.maze.matrices import observation_matrix, priors_vector, c_vector, transitioner
from applications.maze.display import display_qx_text
from agents.discrete_agent import DiscreteAgent
from applications.maze.utils import handle_input
import pygame

def run_maze_game():
    world = MazeWorld(environment=MazeGame(), machina_type='matrix', A=observation_matrix)
    agent = DiscreteAgent(px_vector=priors_vector, c_vector=c_vector, transition=transitioner, machina_type='matrix', A=observation_matrix, q_learning_rate=10)
    clock = pygame.time.Clock()
    
    while True:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # Handle keyboard input
        action = handle_input()
        world.step(action)
        y = world.observe()
        
        # Store previous Qx values
        prev_qx = [float(round(agent.qx.probability(x), 3)) for x in range(15)]
        
        for _ in range(20):
            agent.adjust_q(y)
            
        # Calculate differences
        current_qx = [float(round(agent.qx.probability(x), 3)) for x in range(15)]
        qx_differences = [current - prev for current, prev in zip(current_qx, prev_qx)]
        
        # Draw everything before flipping
        world.display()  # This no longer flips the display
        display_qx_text(world, agent, qx_differences)  # This updates only changed regions
        
        # Single synchronized display update
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(60)

if __name__ == "__main__":
    run_maze_game() 