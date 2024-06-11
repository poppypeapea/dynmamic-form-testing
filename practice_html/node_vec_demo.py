import networkx as nx
from bs4 import BeautifulSoup
from node2vec import Node2Vec
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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

# Convert embeddings to a NumPy array and labels
embedding_vectors = np.array(list(node_embeddings.values()))
labels = list(node_embeddings.keys())

# Reduce dimensions using t-SNE
perplexity = min(embedding_vectors.shape[0] - 1, 10)  # Adjust perplexity to be less than the number of samples
tsne_model = TSNE(n_components=2, perplexity=perplexity, random_state=42)
embeddings_2d = tsne_model.fit_transform(embedding_vectors)

# Visualize the graph
plt.figure(figsize=(10, 10))
plt.title("Node2Vec model Graph Visualization") 
pos = nx.spring_layout(graph, k=0.15)  # Increase k to reduce overlap
nx.draw(graph, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_color="black")
plt.show()

# Visualize the embeddings
plt.figure(figsize=(10, 10))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', edgecolors='k')
for i, label in enumerate(labels):
    plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), textcoords="offset points", xytext=(0, 10), ha='center')
plt.title("t-SNE Visualization of Node2Vec Embeddings")
plt.show()
