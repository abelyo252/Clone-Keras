import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def func(x, y):
    return x ** 2 + y ** 2

def gradient(x, y):
    return np.array([2 * x , 2 * y])

def gradient_descent(x, y, learning_rate):
    grad = gradient(x, y)
    x -= learning_rate * grad[0]
    y -= learning_rate * grad[1]
    return x, y



def gradient_descent_momentum(x, y, learning_rate, momentum, velocity_x, velocity_y):
    grad = gradient(x, y)
    velocity_x = momentum * velocity_x - learning_rate * grad[0]
    velocity_y = momentum * velocity_y - learning_rate * grad[1]
    x += velocity_x
    y += velocity_y
    return x, y, velocity_x, velocity_y

fig = plt.figure(figsize=(12, 6))
ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
ax_contour = fig.add_subplot(1, 2, 2)

x_range = np.linspace(-5, 5, 100)
y_range = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = func(X, Y)

x_start, y_start = 4, -4

learning_rate = 0.1
momentum = 0.9
num_iterations = 100

x_trajectory_gd = [x_start]
y_trajectory_gd = [y_start]
x_trajectory_momentum = [x_start]
y_trajectory_momentum = [y_start]

velocity_x_momentum = 0
velocity_y_momentum = 0

for i in range(num_iterations):
    x_start, y_start = gradient_descent(x_start, y_start, learning_rate)
    x_trajectory_gd.append(x_start)
    y_trajectory_gd.append(y_start)

    x_start, y_start, velocity_x_momentum, velocity_y_momentum = gradient_descent_momentum(x_start, y_start, learning_rate, momentum, velocity_x_momentum, velocity_y_momentum)
    x_trajectory_momentum.append(x_start)
    y_trajectory_momentum.append(y_start)

def update_plot(frame):
    ax_3d.cla()
    ax_contour.cla()

    ax_3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax_3d.scatter(x_trajectory_gd[:frame+1], y_trajectory_gd[:frame+1], func(np.array(x_trajectory_gd[:frame+1]), np.array(y_trajectory_gd[:frame+1])), color='red', s=20, label='Gradient Descent')
    ax_3d.scatter(x_trajectory_momentum[:frame+1], y_trajectory_momentum[:frame+1], func(np.array(x_trajectory_momentum[:frame+1]), np.array(y_trajectory_momentum[:frame+1])), color='blue', s=20, label='Gradient Descent with Momentum')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title('3D Plot')
    ax_3d.legend()

    ax_contour.contourf(X, Y, Z, cmap='viridis', levels=20)
    ax_contour.plot(x_trajectory_gd[:frame+1], y_trajectory_gd[:frame+1], color='red', label='Gradient Descent')
    ax_contour.plot(x_trajectory_momentum[:frame+1], y_trajectory_momentum[:frame+1], color='blue', label='Gradient Descent with Momentum')
    ax_contour.scatter(x_trajectory_gd[frame], y_trajectory_gd[frame], color='red', label='Current Point (GD)')
    ax_contour.scatter(x_trajectory_momentum[frame], y_trajectory_momentum[frame], color='blue', label='Current Point (Momentum)')
    ax_contour.set_xlabel('X')
    ax_contour.set_ylabel('Y')
    ax_contour.set_title(f'Contour Plot (Epoch: {frame+1})')
    ax_contour.legend()

animation = FuncAnimation(fig, update_plot, frames=num_iterations, interval=200)

plt.show()