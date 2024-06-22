import networkx as nx
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.manifold import TSNE
from GraphSAGE import GraphSAGE

# HTML to Graph Conversion Functions
def convert_html_to_graph(html_doc):
    """Convert an HTML document into a graph."""
    soup = BeautifulSoup(html_doc, 'html.parser')
    graph = nx.DiGraph()

    def add_edges(parent, soup_node):
        """Recursively add edges to the graph."""
        for child in soup_node.children:
            if child.name:  # Only consider tag elements
                graph.add_edge(str(parent), str(child))
                add_edges(child, child)

    root = soup.find()
    graph.add_node(str(root))
    add_edges(root, root)
    return graph

def generate_node_features(graph):
    """Generate features for each node in the graph."""
    labels = {
        'div': 0,
        'p': 1,
        'h1': 2,
        'h2': 3,
        'h3': 4,
        'a': 5,
        'li': 6,
        'ul': 7,
        'ol': 8,
        'footer': 9,
        'header': 10,
        'section': 11,
        'article': 12,
        'aside': 13,
        'main': 14,
        'nav': 15,
        'form': 16,
        'input': 17,
        'textarea': 18,
        'button': 19,
        'label': 20,
        'span': 21,
        'img': 22,
        'other': 23
    }
    for node in graph.nodes:
        tag = BeautifulSoup(node, 'html.parser').find().name
        graph.nodes[node]['feature'] = str(node)
        graph.nodes[node]['label'] = labels.get(tag, labels['other'])
    return graph

def nx_to_torch_geometric(graph):
    """Convert a NetworkX graph to a PyTorch Geometric graph."""
    nodes = list(graph.nodes)
    node_mapping = {node: i for i, node in enumerate(nodes)}
    edges = [(node_mapping[src], node_mapping[dst]) for src, dst in graph.edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Create dummy features for each node
    features = torch.eye(len(nodes), dtype=torch.float)
    
    # Use the labels generated in generate_node_features
    labels = torch.tensor([graph.nodes[node]['label'] for node in graph.nodes], dtype=torch.long)

    # Create masks for training, validation, and testing
    train_mask = torch.zeros(len(nodes), dtype=torch.bool)
    val_mask = torch.zeros(len(nodes), dtype=torch.bool)
    test_mask = torch.zeros(len(nodes), dtype=torch.bool)

    # For simplicity, use all nodes for training
    train_mask[:] = True

    data = Data(x=features, edge_index=edge_index, y=labels, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return data

# Ensure file path is correct
file_path = 'practice_html/practice.html' 

# Read HTML file content
with open(file_path, 'r', encoding='utf-8') as file:
    html_doc = file.read()

# Convert HTML to graph
graph = convert_html_to_graph(html_doc)
graph = generate_node_features(graph)

# Convert to PyTorch Geometric graph
data = nx_to_torch_geometric(graph)

# Visualize the original graph
plt.figure(figsize=(10, 10))
plt.title("HTML tree Graph Visualization")
pos = nx.spring_layout(graph, k=0.15)  # Increase k to reduce overlap
nx.draw(graph, pos, with_labels=True, node_size=700, node_color=data.y.numpy(), cmap='viridis', font_size=10, font_color="black")
plt.show()

# Create a GraphSAGE model
model = GraphSAGE(dim_in=data.num_node_features, dim_h=128, dim_out=24)  # Assuming 24 classes for output

# Define a DataLoader
loader = DataLoader([data], batch_size=1)

# Train the model
model.fit(data, loader, epochs=100)

# Get embedding vectors
model.eval()
embeddings = model(data.x, data.edge_index).detach().numpy()


# Visualize the graph with t-SNE
tsne_model = TSNE(n_components=2, perplexity=5, random_state=42)
embeddings_2d = tsne_model.fit_transform(embeddings)

plt.figure(figsize=(10, 10))
plt.title("Node Embeddings Visualized with t-SNE")
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=data.y.numpy(), cmap='viridis', s=20)
plt.colorbar()
plt.show()
