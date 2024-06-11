import networkx as nx
from bs4 import BeautifulSoup
from node2vec import Node2Vec
import numpy as np
import matplotlib.pyplot as plt

def convert_html_to_graph(html_doc):
    soup = BeautifulSoup(html_doc, 'html.parser')
    graph = nx.DiGraph()

    def add_edges(parent, soup_node):
        for child in soup_node.children:
            if child.name:  # Only consider tag elements
                graph.add_edge(str(parent), str(child))
                add_edges(child, child)

    root = soup.find()
    graph.add_node(str(root))
    add_edges(root, root)
    return graph

def generate_node_features(graph):
    for node in graph.nodes:
        graph.nodes[node]['feature'] = str(node)
    return graph


# Ensure file path is correct
file_path = 'practice_html/practice.html'  # If the file is in the current directory, otherwise provide a relative or absolute path

# Read HTML file content
with open(file_path, 'r', encoding='utf-8') as file:
    html_doc = file.read()

# Convert HTML to graph
graph = convert_html_to_graph(html_doc)
# graph = generate_node_features(graph)


# Visualize the graph
plt.figure(figsize=(10, 10))
plt.title("Graph Visualization") 
pos = nx.spring_layout(graph, k=0.15)  # Increase k to reduce overlap
nx.draw(graph, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_color="black")
plt.show()
