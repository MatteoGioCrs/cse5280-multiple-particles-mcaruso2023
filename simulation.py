import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 1. Simulation Parameters ---
N_PARTICLES = 50         
ALPHA = 0.05            
GOAL = np.array([8.0, 8.0])
R_PERSONAL = 1.0        

positions = np.random.rand(N_PARTICLES, 2) * 3
velocities = np.zeros((N_PARTICLES, 2))

def get_wall_gradient(pos, w_start, w_end, R_wall=0.5):
    """Calculates the quadratic penalty gradient from a line segment (wall)."""
    wall_vec = w_end - w_start
    wall_len_sq = np.dot(wall_vec, wall_vec)
    
    if wall_len_sq == 0:
        closest_point = w_start
    else:
        # Project particle position onto the wall segment and clamp between 0 and 1
        t = max(0.0, min(1.0, np.dot(pos - w_start, wall_vec) / wall_len_sq))
        closest_point = w_start + t * wall_vec
        
    diff = pos - closest_point
    dist = np.linalg.norm(diff)
    
    # Apply quadratic band penalty if within R_wall
    if 0 < dist < R_wall:
        return -(R_wall - dist) * (diff / dist)
    return np.zeros(2)

def compute_gradients(pos, vels, beta=2.0):
    gradients = np.zeros_like(pos)
    
    for i in range(N_PARTICLES):
        xi = pos[i]
        grad_i = np.zeros(2)
        
        # Determine the unit direction vector of motion (v_hat)
        speed = np.linalg.norm(vels[i])
        if speed > 0.001:
            v_hat = vels[i] / speed
        else:
            # Fallback: If stationary, assume facing the goal
            v_hat = (GOAL - xi)
            v_hat = v_hat / np.linalg.norm(v_hat)

        # A. Goal Gradient (Same as before)
        grad_goal = (xi - GOAL)
        grad_i += 0.1 * grad_goal 
        
        # B. Social Force Gradient (Anisotropic)
        for j in range(N_PARTICLES):
            if i == j: continue
            
            xj = pos[j]
            diff = xi - xj 
            dij = np.linalg.norm(diff)
            
            if 0 < dij <= R_PERSONAL:
                # 1. Calculate the standard isotropic gradient
                grad_iso = -(R_PERSONAL - dij) * (diff / dij)
                
                # 2. Calculate the anisotropic directional weight
                direction_to_j = (xj - xi) / dij
                dot_prod = np.dot(v_hat, direction_to_j)
                weight = 1.0 + beta * max(0.0, dot_prod)

                # 3. Apply the weight to the social gradient
                grad_i += grad_iso * weight
                
        # C. Wall Gradient
        walls = [
            (np.array([2.0, 0.0]), np.array([2.0, 5.0])),
            (np.array([5.0, 5.0]), np.array([5.0, 10.0]))
        ]
        
        # C. Wall Gradient
        for w_start, w_end in walls:
            grad_i += 50.0 * get_wall_gradient(xi, w_start, w_end)
        
        gradients[i] = grad_i
        
    return gradients

# --- 2. Animation Setup ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1, 10)
ax.set_ylim(-1, 10)
ax.set_title("Multi-Particle Animation (Isotropic)")

# Draw the walls on the plot
walls = [
    (np.array([2.0, 0.0]), np.array([2.0, 5.0])),
    (np.array([5.0, 5.0]), np.array([5.0, 10.0]))
]
# Draw the walls on the plot
for w_start, w_end in walls:
    # Extracts the X's for the first list, and Y's for the second
    ax.plot([w_start[0], w_end[0]], [w_start[1], w_end[1]], 'k-', linewidth=3)

# --- Generate the Visual Gradient (Background Potential Field) ---
# Create a 100x100 grid spanning your map coordinates
x_vals = np.linspace(-1, 10, 100)
y_vals = np.linspace(-1, 10, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.zeros_like(X)

R_wall = 0.5 # Must match the radius in your wall gradient function

# Calculate the static cost (Goal + Walls) for every point on the grid
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = np.array([X[i, j], Y[i, j]])
        
        # 1. Goal Cost (0.5 * ||x - goal||^2) scaled by your goal weight
        cost = 0.05 * 0.5 * np.linalg.norm(point - GOAL)**2 
        
        # 2. Wall Cost
        for w_start, w_end in walls:
            wall_vec = w_end - w_start
            wall_len_sq = np.dot(wall_vec, wall_vec)
            if wall_len_sq > 0:
                t = max(0.0, min(1.0, np.dot(point - w_start, wall_vec) / wall_len_sq))
                closest = w_start + t * wall_vec
                dist = np.linalg.norm(point - closest)
                
                # If within the wall's radius, add the quadratic penalty
                if 0 < dist < R_wall:
                    # Scaled by your wall multiplier (e.g., 50.0)
                    cost += 50.0 * 0.5 * (R_wall - dist)**2 
                    
        Z[i, j] = cost

# Draw the contour gradient map
# cmap='viridis' gives a nice purple-to-yellow gradient; alpha makes it slightly transparent
contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.5)
fig.colorbar(contour, ax=ax, label='Potential Field Cost')

# Plot goal and initial particles
ax.plot(GOAL[0], GOAL[1], 'rX', markersize=10, label='Goal')
scatter = ax.scatter(positions[:, 0], positions[:, 1], c='blue', s=50, label='Particles')
ax.legend()

def update(frame):
    global positions, velocities
    
    # Calculate total gradient (pass velocities now)
    grads = compute_gradients(positions, velocities, beta=0.0)
    
    # Calculate new positions
    new_positions = positions - ALPHA * grads
    
    # Update velocities (displacement over this step)
    velocities = new_positions - positions
    positions = new_positions
    
    scatter.set_offsets(positions)
    return scatter,

# --- 3. Run Simulation ---
# Generate animation over time [cite: 93]
ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=True)

plt.show()