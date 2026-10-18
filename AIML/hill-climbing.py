import matplotlib.pyplot as plt
import numpy as np

def hill_climbing(function, start, step_size, max_iterations):
    current = start
    path = [current]  # Keep track of the path for plotting
    
    for _ in range(max_iterations):
        left = current - step_size
        right = current + step_size
        
        if function(left) > function(current):
            current = left
        elif function(right) > function(current):
            current = right
        else:
            break  # Local peak reached
            
        path.append(current)
        
    return current, function(current), path

# We want to maximize: f(x) = -(x-3)^2 + 9
def objective(x):
    return -(x - 3) ** 2 + 9

start = 0
step_size = 0.5
max_iterations = 100

best_x, best_value, path = hill_climbing(objective, start, step_size, max_iterations)
print("Best Solution (x):", best_x)
print("Maximum Value:", best_value)

# --- Visualization ---
# Generate x values for the smooth curve of the objective function
x_vals = np.linspace(-1, 7, 400)
y_vals = objective(x_vals)

# Extract y-values for the path taken by the algorithm
path_x = np.array(path)
path_y = objective(path_x)

plt.figure(figsize=(10, 6))

# Plot the objective function
plt.plot(x_vals, y_vals, label="Objective Function: $f(x) = -(x-3)^2 + 9$", color="royalblue", linewidth=2)

# Plot the hill-climbing path steps
plt.plot(path_x, path_y, color="darkorange", marker="o", linestyle="--", linewidth=1.5, label="Hill Climbing Path")

# Highlight start and end points
plt.scatter(path[0], objective(path[0]), color="green", s=100, zorder=5, label=f"Start (x={path[0]})")
plt.scatter(best_x, best_value, color="red", s=100, zorder=5, label=f"Best Found (x={best_x})")

# Graph styling
plt.title("Hill Climbing Optimization Visualization", fontsize=14, fontweight="bold")
plt.xlabel("x", fontsize=12)
plt.ylabel("f(x)", fontsize=12)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="lower center")

plt.show()