from agents.demo_agent import DemoAgent
from applications.demo.world import DemoWorld
from applications.demo.plot.InteractivePlot import InteractivePlot
import matplotlib.pyplot as plt

def run_quadratic_demo():
    # Initialize agent and world
    # Quadratic machina
    world = DemoWorld(machina_type='linear', b1=2.0, b0=4.0)
    agent = DemoAgent(machina_type='linear', b1=2.0, b0=4.0)
    
    # Create interactive plot
    plot = InteractivePlot(agent, world, vfe=True, complexity=True, accuracy=True, real_x=True, min_vfe=True, current_mu=True)
    plt.show()

if __name__ == "__main__":
    run_quadratic_demo() 