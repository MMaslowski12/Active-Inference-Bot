import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from core.distributions import Normal
from core.machinas import LinearMachina as Linear, QuadraticMachina as Quadratic

class InteractivePlot:
    def __init__(self, agent, world, vfe=True, complexity=True, 
                 accuracy=True, real_x=True, min_vfe=True, current_mu=True,
                 machina_graph=True):
        self.agent = agent
        self.world = world
        # Store plot visibility flags
        self.vfe = vfe
        self.complexity = complexity
        self.accuracy = accuracy
        self.real_x = real_x
        self.min_vfe = min_vfe
        self.current_mu = current_mu
        self.machina_graph = machina_graph
        
        # Create figure with one or two subplots based on machina_graph
        if self.machina_graph:
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(20, 12))
        else:
            self.fig, self.ax1 = plt.subplots(figsize=(10, 12))
            self.ax2 = None
        
        plt.subplots_adjust(bottom=0.3)  # Increased bottom margin to make room for x-labels
        
        # Button dimensions and positions
        button_width = 0.15
        button_height = 0.075
        top_row_y = 0.15
        bottom_row_y = 0.05
        
        # Top row x-positions
        top_x1 = 0.1
        top_x2 = 0.3
        top_x3 = 0.5
        top_x4 = 0.7
        
        # Calculate bottom row x-positions (centered in the actual gaps)
        # For each gap: right edge of left button to left edge of right button
        bottom_x1 = ((top_x1 + button_width) + top_x2) / 2 - button_width / 2
        bottom_x2 = ((top_x2 + button_width) + top_x3) / 2 - button_width / 2
        bottom_x3 = ((top_x3 + button_width) + top_x4) / 2 - button_width / 2
        
        # First row of buttons
        self.ax_prev = plt.axes([top_x1, top_row_y, button_width, button_height])
        self.ax_next = plt.axes([top_x2, top_row_y, button_width, button_height])
        self.ax_var_down = plt.axes([top_x3, top_row_y, button_width, button_height])
        self.ax_var_up = plt.axes([top_x4, top_row_y, button_width, button_height])
        
        # Second row of buttons
        self.ax_grad = plt.axes([bottom_x1, bottom_row_y, button_width, button_height])
        self.ax_learn_px = plt.axes([bottom_x2, bottom_row_y, button_width, button_height])
        self.ax_learn_py_x = plt.axes([bottom_x3, bottom_row_y, button_width, button_height])
        
        # Create all buttons
        self.btn_prev = Button(self.ax_prev, 'Previous State')
        self.btn_next = Button(self.ax_next, 'Next State')
        self.btn_var_down = Button(self.ax_var_down, 'Decrease σ(q)')
        self.btn_var_up = Button(self.ax_var_up, 'Increase σ(q)')
        self.btn_grad = Button(self.ax_grad, 'Adjust q(x)')
        self.btn_learn_px = Button(self.ax_learn_px, 'Learn p(x)')
        self.btn_learn_py_x = Button(self.ax_learn_py_x, 'Learn p(y|x)')
        
        # Connect buttons to functions
        self.btn_prev.on_clicked(self.prev_state)
        self.btn_next.on_clicked(self.next_state)
        self.btn_var_down.on_clicked(self.decrease_variance)
        self.btn_var_up.on_clicked(self.increase_variance)
        self.btn_grad.on_clicked(self.gradient_step_mu)
        self.btn_learn_px.on_clicked(self.learn_px)
        self.btn_learn_py_x.on_clicked(self.learn_py_x)
        
        # Initial plot
        self.update_plot()
    
    def gradient_step_mu(self, event):
        y = self.world.observe()
        self.agent.adjust_q(y)
        self.update_plot()
    
    def prev_state(self, event):
        self.world.step(-1)
        self.update_plot()
    
    def next_state(self, event):
        self.world.step(1)
        self.update_plot()
    
    def decrease_variance(self, event):
        current_std = self.agent.qx.std
        self.agent.qx.std = max(0.1, current_std * 0.8)  # Minimum std of 0.1
        self.update_plot()
    
    def increase_variance(self, event):
        current_std = self.agent.qx.std
        self.agent.qx.std = min(5.0, current_std * 1.2)  # Maximum std of 5.0
        self.update_plot()
    
    def learn_px(self, event):
        y = self.world.observe()
        self.agent.learn_px(y)
        self.update_plot()
    
    def learn_py_x(self, event):
        y = self.world.observe()
        self.agent.learn_py_x(y)
        self.update_plot()
    
    def update_plot(self):
        # Clear plots
        self.ax1.clear()
        if self.machina_graph:
            self.ax2.clear()
        
        # First subplot: VFE and components
        q_mu_values = np.linspace(-5, 5, 200)
        vfe_values = []
        complexity_values = []
        accuracy_values = []
        
        # Store current q values
        old_q_mu = self.agent.qx.mean
        old_q_std = self.agent.qx.std
        
        # Get observation once
        y = self.world.observe()
        
        # Calculate VFE for each q_mu using Agent's methods
        for q_mu in q_mu_values:
            self.agent.qx.mean = q_mu  # Temporarily set q's mean
            complexity = self.agent.calculate_complexity()
            accuracy = self.agent.calculate_accuracy(y)
            vfe = self.agent.calculate_vfe(y)
            
            complexity_values.append(complexity)
            accuracy_values.append(accuracy)
            vfe_values.append(vfe)
        
        # Restore original q values
        self.agent.qx.mean = old_q_mu
        self.agent.qx.std = old_q_std
        
        # Find minimum VFE point
        min_vfe_idx = np.argmin(vfe_values)
        min_vfe_mu = q_mu_values[min_vfe_idx]
        min_vfe = vfe_values[min_vfe_idx]
        
        # Plot curves based on visibility flags
        if self.vfe:
            self.ax1.plot(q_mu_values, vfe_values, label='VFE', color='black')
        if self.complexity:
            self.ax1.plot(q_mu_values, complexity_values, label='Complexity', color='red', linestyle='--')
        if self.accuracy:
            self.ax1.plot(q_mu_values, accuracy_values, label='Negative Accuracy', color='blue', linestyle='--')
        
        # Add vertical lines based on visibility flags
        real_x = self.world._get_state()
        if self.real_x:
            self.ax1.axvline(x=real_x, color='green', linestyle=':', label=f'Real x = {real_x}')
        if self.min_vfe:
            self.ax1.axvline(x=min_vfe_mu, color='purple', linestyle=':', label=f'Min VFE x = {min_vfe_mu:.2f}')
            # Add a point at the minimum
            self.ax1.plot(min_vfe_mu, min_vfe, 'ro', label=f'Min VFE = {min_vfe:.2f}')
        
        # Add current q_mu and q_sigma values
        current_mu = self.agent.qx.mean
        current_sigma = self.agent.qx.std
        if self.current_mu:
            self.ax1.axvline(x=current_mu, color='orange', linestyle='-', label=f'Current μ = {current_mu:.2f}')
        
        # Add p(x) mean line
        px_mean = self.agent.px.mean
        self.ax1.axvline(x=px_mean, color='cyan', linestyle='--', label=f'p(x) μ = {px_mean:.2f}')
        
        self.ax1.set_xlabel('q_mu')
        self.ax1.set_ylabel('Value')
        self.ax1.set_title(f'VFE and its Components (World State: {real_x}, σ: {current_sigma:.2f})')
        self.ax1.grid(True)
        self.ax1.legend()
        
        # Second subplot: Machina graphs (only if enabled)
        if self.machina_graph:
            x_values = np.linspace(-5, 5, 200)
            world_machina = [self.world._machina(x) for x in x_values]
            agent_machina = [self.agent.py_x(x).mean for x in x_values]  # Use py_x to get the mean of the conditional distribution
            
            # Get current parameter values and format equations based on machina type
            if isinstance(self.world._machina, Linear):
                world_b1 = self.world._machina.b1
                world_b0 = self.world._machina.b0
                world_eq = f'y = {world_b1:.2f}x + {world_b0:.2f}'
            else:  # Quadratic
                world_a = self.world._machina.a
                world_b = self.world._machina.b
                world_c = self.world._machina.c
                world_eq = f'y = {world_a:.2f}x² + {world_b:.2f}x + {world_c:.2f}'
            
            # For agent, get parameters from the underlying machina in py_x
            if isinstance(self.agent.py_x.machina, Linear):
                agent_b1 = self.agent.py_x.machina.b1
                agent_b0 = self.agent.py_x.machina.b0
                agent_eq = f'y = {agent_b1:.2f}x + {agent_b0:.2f}'
            else:  # Quadratic
                agent_a = self.agent.py_x.machina.a
                agent_b = self.agent.py_x.machina.b
                agent_c = self.agent.py_x.machina.c
                agent_eq = f'y = {agent_a:.2f}x² + {agent_b:.2f}x + {agent_c:.2f}'
            
            # Plot lines with simpler legend labels
            self.ax2.plot(x_values, world_machina, label='World', color='blue')
            self.ax2.plot(x_values, agent_machina, label='Agent', color='red', linestyle='--')
            
            # Add equation labels next to lines at different positions
            self.ax2.text(8, world_machina[-1] - 2, world_eq, 
                         color='blue', ha='right', va='top')
            self.ax2.text(8, agent_machina[-1] + 2, agent_eq, 
                         color='red', ha='right', va='bottom')
            
            self.ax2.axvline(x=real_x, color='green', linestyle=':', label=f'Real x = {real_x}')
            self.ax2.axvline(x=current_mu, color='orange', linestyle='-', label=f'Current μ = {current_mu:.2f}')
            
            # Mark intersections
            world_y_at_real_x = self.world._machina(real_x)
            agent_y_at_current_mu = self.agent.py_x(current_mu).mean  # Use py_x to get the mean
            
            self.ax2.plot(real_x, world_y_at_real_x, 'go', label=f'World y at x={real_x}')
            self.ax2.plot(current_mu, agent_y_at_current_mu, 'ro', label=f'Agent y at μ={current_mu:.2f}')
            
            self.ax2.set_xlabel('x')
            self.ax2.set_ylabel('y')
            self.ax2.set_title('Machina Functions')
            self.ax2.grid(True)
            self.ax2.legend()
        
        # Force a redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()