import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


# Define the 3D function
def function(x, y):
    return np.sin(x) * np.sin(y) + 0.1 * (x ** 2 + y ** 2)


# Calculate the gradient of the function
def gradient(x, y):
    dx = np.cos(x) * np.sin(y) + 0.2 * x
    dy = np.sin(x) * np.cos(y) + 0.2 * y
    return dx, dy


# Perform gradient descent optimization
def gradient_descent(lr, num_steps):
    x = -4.0  # Initial x-coordinate
    y = 4.0  # Initial y-coordinate

    path = [(x, y)]  # Store the path taken during optimization

    for epoch in range(num_steps):
        df_dx, df_dy = gradient(x, y)
        x -= lr * df_dx
        y -= lr * df_dy
        path.append((x, y))

    return path


# Create the figure and subplots
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)

# Define the range for x and y
x_range = np.linspace(-5, 5, 100)
y_range = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = function(X, Y)

# Plot the 3D function
ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('3D Function')

# Create an empty contour plot
contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Contour Plot')

# Initialize the scatter plot and trajectory lines for the optimization path
scatter1 = ax1.scatter([], [], [], c='red')
scatter2 = ax2.scatter([], [], c='red')
trajectory1, = ax1.plot([], [], [], c='blue', linestyle='dashed')
trajectory2, = ax2.plot([], [], c='blue', linestyle='dashed')

# Initialize the optimization path
path = gradient_descent(lr=0.1, num_steps=50)

# Initialize the epoch indicator text
epoch_text1 = ax1.text(-5, 5, function(-5, 5) + 5, '', fontsize=10, color='red')


# Update function for the animation
def update(frame):
    scatter1._offsets3d = ([path[frame][0]], [path[frame][1]], [function(*path[frame])])
    scatter2.set_offsets([path[frame]])
    epoch_text1.set_text(f'Epoch: {frame + 1}')

    ax1.set_title('3D Function - Gradient Descent Optimization')
    ax2.set_title('Contour Plot - Gradient Descent Optimization')

    # Update trajectory
    trajectory1.set_data(*zip(*path[:frame+1]))
    trajectory1.set_3d_properties([function(*point) for point in path[:frame+1]])
    trajectory2.set_data(*zip(*path[:frame+1]))

    return scatter1, scatter2, epoch_text1 , trajectory1, trajectory2

# Create the animation
animation = FuncAnimation(fig, update, frames=len(path), interval=500, blit=True)
animation.save("grad.gif",writer="ffmpeg")

# Display the plot
plt.show()