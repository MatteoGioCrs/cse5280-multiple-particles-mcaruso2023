import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 1. Simulation Parameters ---
N_PARTICLES = 10        # Number of agents 
ALPHA = 0.05            # Step size for gradient descent [cite: 57]
GOAL = np.array([8.0, 8.0])
R_PERSONAL = 1.0        # Radius for personal space (Isotropic force) [cite: 68]

# Initialize N particles with random distinct starting positions 
positions = np.random.rand(N_PARTICLES, 2) * 3

def compute_gradients(pos):
    """Calculates the total gradient for each particle."""
    gradients = np.zeros_like(pos)
    
    for i in range(N_PARTICLES):
        xi = pos[i]
        grad_i = np.zeros(2)
        
        # A. Goal Gradient (Attractive Cost)
        # Cost = 0.5 * ||xi - GOAL||^2  =>  Gradient = xi - GOAL
        grad_goal = (xi - GOAL)
        grad_i += 0.1 * grad_goal  # Weighting the goal attraction
        
        # B. Social Force Gradient (Isotropic - Quadratic Repulsion) [cite: 67]
        for j in range(N_PARTICLES):
            if i == j:
                continue
            xj = pos[j]
            diff = xi - xj
            dij = np.linalg.norm(diff) # Distance between particle i and j [cite: 69]
            
            # If within personal space, apply repulsion [cite: 68]
            if 0 < dij <= R_PERSONAL:
                # Gradient of C_social = 0.5 * (R - dij)^2  => -(R - dij) * (diff / dij)
                grad_social = -(R_PERSONAL - dij) * (diff / dij)
                grad_i += grad_social
                
        # C. Wall Gradient
        # TODO: Implement wall penalty terms based on your floor plan [cite: 47, 53]
        
        gradients[i] = grad_i
        
    return gradients

# --- 2. Animation Setup --- 
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1, 10)
ax.set_ylim(-1, 10)
ax.set_title("Multi-Particle Animation (Isotropic)")

# Plot goal and initial particles
ax.plot(GOAL[0], GOAL[1], 'rX', markersize=10, label='Goal')
scatter = ax.scatter(positions[:, 0], positions[:, 1], c='blue', s=50, label='Particles')
ax.legend()

def update(frame):
    """Animation update function called every frame."""
    global positions
    
    # Calculate total gradient for all particles
    grads = compute_gradients(positions)
    
    # Update positions via Gradient Descent: x^(k+1) = x^(k) - alpha * grad(C) 
    positions = positions - ALPHA * grads
    
    # Update the scatter plot data
    scatter.set_offsets(positions)
    return scatter,

# --- 3. Run Simulation ---
# Generate animation over time [cite: 93]
ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=True)

plt.show()