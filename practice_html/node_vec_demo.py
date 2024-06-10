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

def generate_embeddings(graph, dimensions=128, walk_length=30, num_walks=200, workers=4):
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    return model

# Ensure file path is correct
file_path = 'practice_html/practice.html'  # If the file is in the current directory, otherwise provide a relative or absolute path

# Read HTML file content
with open(file_path, 'r', encoding='utf-8') as file:
    html_doc = file.read()

# Convert HTML to graph
graph = convert_html_to_graph(html_doc)
graph = generate_node_features(graph)

# Generate node embeddings
model = generate_embeddings(graph)

# Get embedding vectors, ensuring the node name is in the model's vocabulary
node_embeddings = {node: model.wv[str(node)] for node in graph.nodes if str(node) in model.wv}

# Print node embeddings
print(node_embeddings)

# Visualize the graph
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(graph)
nx.draw(graph, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_color="black")
plt.title("Graph Visualization")
plt.show()
