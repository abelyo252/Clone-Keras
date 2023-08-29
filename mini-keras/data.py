import math

class Data:

    def __init__(self, data, _children=(), _op='', label=''):
        # Initialize a Data instance with a numerical value, children nodes, operation, and label
        self.data = data
        self._children = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0  # Gradient associated with this node
        # a "node" refers to an instance of the `Data` class.
        # Each `Data` instance represents a node in the computational graph.
        #

    def __repr__(self):
        return f"Data(data={self.data})"

    def __add__(self, other):
        # Overload the '+' operator to perform addition between Data instances
        out = Data(self.data + other.data, (self, other), '+')

        def _backward():
            # Backpropagation: Accumulate gradients for addition operation
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        # Overload the '*' operator to perform multiplication between Data instances
        out = Data(self.data * other.data, (self, other), '*')

        def _backward():
            # Backpropagation: Accumulate gradients for multiplication operation
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        # Apply the hyperbolic tangent activation function to the node's value
        t = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        out = Data(t, (self,), 'tanh')

        def _backward():
            # Backpropagation: Accumulate gradients for the tanh operation
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    @staticmethod
    def backward(output):
        # Perform backward propagation to compute gradients
        output.grad = 1.0  # Set the gradient of the output node to 1.0
        seen = set()  # Set to keep track of visited nodes during backward propagation
        stack = [output]  # Stack to store nodes for traversal

        while stack:
            node = stack.pop()
            if node not in seen:
                seen.add(node)  # Mark the node as visited
                stack.extend(node._children)  # Add children nodes to the stack for traversal
                node._backward()  # Call the _backward method of the node to accumulate gradients


    def _backward(self):
        # Accumulate gradients for child nodes during backward propagation
        for child in self._children:
            child.grad += self.grad
