"""
Mathematical expressions with multiple local minima and one absolute minimum.
Here are a few examples:

1. Polynomial Functions:
   - `f(x) = x^4 - 4x^2 + x` (used in the previous example)
   - `f(x) = x^6 - 5x^4 + 4x^3 + 3x^2 - 2x`
   - `f(x) = x^5 - 6x^3 + 9x`

2. Trigonometric Functions:
   - `f(x) = sin(x) + sin(2x) + sin(3x)`
   - `f(x) = sin(2x) - cos(3x) + sin(4x)`

3. Exponential Functions:
   - `f(x) = e^x - 2e^(-2x) + e^(-3x)`
   - `f(x) = e^(-x) + e^(-2x) + e^(-3x) + e^(-4x)`

4. Combination of Functions:
   - `f(x) = x^4 - 4x^3 + 3x^2 + sin(2x) + 2cos(3x)`


"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def function(x):
    return np.sin(x)

def gradient(x):
    return np.cos(x)

def gradient_descent(gradient_func, initial_x, learning_rate, num_iterations):
    x_values = [initial_x]
    for _ in range(num_iterations):
        gradient_value = gradient_func(x_values[-1])
        new_x = x_values[-1] - learning_rate * gradient_value
        x_values.append(new_x)
    return x_values

def gradient_descent_momentum(gradient_func, initial_x, learning_rate, momentum, num_iterations):
    x_values = [initial_x]
    velocity = 0
    for _ in range(num_iterations):
        gradient_value = gradient_func(x_values[-1])
        velocity = momentum * velocity - learning_rate * gradient_value
        new_x = x_values[-1] + velocity
        x_values.append(new_x)
    return x_values

x = np.linspace(-2*np.pi, 2*np.pi, 100)
y = function(x)
absolute_min_x = x[np.argmin(y)]
absolute_min_y = np.min(y)

fig, ax = plt.subplots()
ax.plot(x, y)
plt.scatter(absolute_min_x, absolute_min_y, color='green', label='Absolute Minimum')
scatter_gd = ax.scatter([], [], color='red', label='Gradient Descent')
scatter_momentum = ax.scatter([], [], color='blue', label='Gradient Descent with Momentum')
epoch_text = ax.text(0.98, 0.02, '', transform=ax.transAxes, color='black', ha='right', va='bottom')

ax.set_xlabel('X')
ax.set_ylabel('f(X)')
ax.set_title('Function with Multiple Local Minima')
ax.legend()

initial_X = np.pi/2+0.2

def update(frame):
    iteration = frame + 1
    x_values_gd = gradient_descent(gradient, initial_X, 0.1, iteration)
    y_values_gd = function(np.array(x_values_gd))
    scatter_gd.set_offsets(np.column_stack((x_values_gd, y_values_gd)))

    x_values_momentum = gradient_descent_momentum(gradient, initial_X, 0.1, 0.9, iteration)
    y_values_momentum = function(np.array(x_values_momentum))
    scatter_momentum.set_offsets(np.column_stack((x_values_momentum, y_values_momentum)))

    epoch_text.set_text('Epoch: {}/{}'.format(iteration, num_iterations))
    return scatter_gd, scatter_momentum, epoch_text

num_iterations = 100
animation = FuncAnimation(fig, update, frames=num_iterations, interval=200, blit=True)

plt.show()