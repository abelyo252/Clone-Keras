import networkx as nx
import pygraphviz as pgv
from PIL import Image as PilImage
import numpy as np
import io
from IPython.display import Image as IpyImage


def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._children:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges


def trace(root):
    """
          Builds a set of all nodes and edges in a graph by performing a depth-first traversal.
          Args:
              root: The root node of the computational graph.
          Returns:
              nodes: Set of nodes in the graph.
              edges: Set of edges in the graph.
    """
    nodes, edges = set(), set() # Initialize empty sets for nodes and edges

    def build(v):
        """
            Recursive helper function to build the set of nodes and edges in a graph.
            Args:
                v: Current node being visited.
       """
        if v not in nodes:  # Check if the node has not been added to the set of nodes
            nodes.add(v)  # Add the node to the set of nodes
            for child in v._children:  # Iterate over the children of the current node
                edges.add((child, v))  # Add an edge from the child to the current node
                build(child) # Recursively build the graph for the child node

    build(root) # Start building the graph from the root node
    return nodes, edges # Return the set of nodes and edges of the graph


def visualize_computational_graph(output_node):
    """
    Creates a graphical representation of a computational graph.
    Args:
        output_node: The output node of the computational graph.
    Returns:
        viz1_array: Array representation of the rendered graph.
    """
    dot = nx.DiGraph(format='png', graph_attr={'rankdir': 'LR', 'bgcolor': 'transparent'})  # LR = left to right
    nodes, edges = trace(output_node)
    node_mapping = {}  # Mapping of string IDs to node objects

    # Add nodes to the graph
    for n in nodes:
        uid = str(id(n))
        node_mapping[uid] = n  # Store the mapping of ID to node object
        label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad)
        dot.add_node(uid, label=label, shape='record', fontname='Arial', fontsize=10, style='filled',
                     fillcolor='lightblue', color='black')
        if n._op:
            op_uid = uid + n._op
            node_mapping[op_uid] = n  # Store the mapping of ID with operation name to node object
            dot.add_node(op_uid, label=n._op, fontname='Arial', fontsize=10, shape='box', style='filled',
                         fillcolor='lightgray')
            dot.add_edge(op_uid, uid)

    # Add edges to the graph
    for n1, n2 in edges:
        dot.add_edge(str(id(n1)), str(id(n2)) + n2._op)

    pos = nx.spring_layout(dot, seed=42)  # Horizontal layout
    labels = {n: "{:.4f}".format(node_mapping[n].data) for n in dot.nodes()}
    edge_labels = {(u, v): labels[v] for u, v in dot.edges()}

    # Create a PyGraphviz graph from NetworkX graph
    pgv_graph = nx.nx_agraph.to_agraph(dot)

    # Set visual attributes for the graph nodes and edges
    pgv_graph.graph_attr.update({'splines': 'polyline'})
    pgv_graph.node_attr.update({'shape': 'box', 'style': 'filled', 'fillcolor': 'lightgray', 'fontname': 'Arial',
                                'fontsize': 10, 'color': 'black'})
    pgv_graph.edge_attr.update({'fontsize': 8})

    # Render the graph to a file (e.g., in PNG format)
    output_file = "computational_graph.png"
    pgv_graph.draw(output_file, prog="dot", format="png")

    # Convert the rendered graph image to an array
    img = IpyImage(filename=output_file)
    viz1_array = np.array(PilImage.open(io.BytesIO(img.data)))

    # Return the array representation of the rendered graph
    return viz1_array


def concatenate_images(image1, image2, axis='horizontal'):
    """
    Concatenates two NumPy images either horizontally or vertically.

    Parameters:
        image1 (numpy.ndarray): First image as a NumPy array.
        image2 (numpy.ndarray): Second image as a NumPy array.
        axis (str, optional): Concatenation axis. 'horizontal' for horizontal concatenation,
                              'vertical' for vertical concatenation. Defaults to 'horizontal'.

    Returns:
        numpy.ndarray: Concatenated image as a NumPy array.
    """
    if axis == 'horizontal':
        return np.concatenate((image1, image2), axis=1)
    elif axis == 'vertical':
        return np.concatenate((image1, image2), axis=0)
    else:
        raise ValueError("Invalid axis parameter. Please use 'horizontal' or 'vertical'.")