import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define the 3D function
def function(x, y):
    return x ** 2 + y ** 2

# Define the gradient of the function
def gradient(x, y):
    dx = 2 * x
    dy = 2 * y
    return dx, dy

# Define the gradient descent with momentum optimization
def gradient_descent_with_momentum(x_start, y_start, learning_rate=0.01, momentum=0.9, num_iterations=100):
    x = x_start
    y = y_start
    velocity_x = 0
    velocity_y = 0
    trajectory = [(x, y)]

    for _ in range(num_iterations):
        dx, dy = gradient(x, y)

        velocity_x = momentum * velocity_x - learning_rate * dx
        velocity_y = momentum * velocity_y - learning_rate * dy

        x += velocity_x
        y += velocity_y

        trajectory.append((x, y))

    return np.array(trajectory)

# Define the range of x and y values
x_vals = np.linspace(-5, 5, 100)
y_vals = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = function(X, Y)

# Set up the figure and subplots
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2)

# Create a 3D plot of the function
ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('3D Function')

# Create a contour plot of the function
contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Contour Plot')

# Initialize the trajectory lines for the plots
trajectory_line_3d, = ax1.plot([], [], [], 'r', label='Gradient Descent with Momentum')
trajectory_line_contour, = ax2.plot([], [], 'r', label='Gradient Descent with Momentum')

# Initialize the point along the trajectory
point_3d = ax1.plot([2], [-2], [function(2, -2)], 'bo', markersize=5, label='Current Point')[0]
point_contour = ax2.plot([2], [-2], 'bo', markersize=5, label='Current Point')[0]

# Perform gradient descent with momentum optimization
trajectory = gradient_descent_with_momentum(x_start=2, y_start=-2, learning_rate=0.01, momentum=0.9, num_iterations=100)

# Function to update the animation
def update_animation(frame):
    x, y = trajectory[frame]
    trajectory_line_3d.set_data(trajectory[:frame, 0], trajectory[:frame, 1])
    trajectory_line_3d.set_3d_properties([function(x, y) for x, y in trajectory[:frame]])
    trajectory_line_contour.set_data(trajectory[:frame, 0], trajectory[:frame, 1])
    point_3d.set_data([x], [y])
    point_3d.set_3d_properties([function(x, y)])
    point_contour.set_data([x], [y])
    return trajectory_line_3d, trajectory_line_contour, point_3d, point_contour

# Create the animation
animation = FuncAnimation(fig, update_animation, frames=len(trajectory), interval=100, blit=True)

# Add legends to the plots
ax1.legend()
ax2.legend()

plt.tight_layout()
plt.show()