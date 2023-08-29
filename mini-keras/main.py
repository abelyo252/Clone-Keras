from data import Data
from visualization import visualize_computational_graph,concatenate_images
import matplotlib.pyplot as plt

# inputs x1,x2
x1 = Data(2.0, label='x1')
x2 = Data(0.0, label='x2')
# weights w1,w2
w1 = Data(-3.0, label='w1')
w2 = Data(1.0, label='w2')
# bias of the neuron
b = Data(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
y = x1w1x2w2 + b; y.label = 'y'

# visualized L graph now
viz_Y = visualize_computational_graph(y)
Data.backward(y)
viz_back_y = visualize_computational_graph(y) # visualized final grad of all object

# Concatenate the images horizontally
concatenated_image = concatenate_images(viz_Y, viz_back_y, axis='horizontal')


# Plot the function and its derivative
plt.figure(figsize=(8, 6))
plt.title("Backpropagation on Linear Regression of x1*w1 + x2*w2 + b")
plt.axis('off')
plt.imshow(concatenated_image)
plt.show()