import networkx as nx
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    """GraphSAGE"""
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.sage1 = SAGEConv(dim_in, dim_h)
        self.sage2 = SAGEConv(dim_h, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)

    def forward(self, x, edge_index):
        h = self.sage1(x, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.sage2(h, edge_index)
        return h

    def fit(self, data, train_loader, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = self.optimizer

        self.train()
        for epoch in range(epochs + 1):
            total_loss = 0
            acc = 0
            val_loss = 0
            val_acc = 0

            # Train on batches
            for batch in train_loader:
                optimizer.zero_grad()
                out = self(batch.x, batch.edge_index)
                loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
                total_loss += loss.item()
                acc += accuracy(out[batch.train_mask].argmax(dim=1), batch.y[batch.train_mask])
                loss.backward()
                optimizer.step()

                # Validation
                val_loss += criterion(out[batch.val_mask], batch.y[batch.val_mask])
                val_acc += accuracy(out[batch.val_mask].argmax(dim=1), batch.y[batch.val_mask])

            # Print metrics every 10 epochs
            if epoch % 10 == 0:
                print(f'Epoch {epoch:>3} | Train Loss: {total_loss / len(train_loader):.3f} '
                      f'| Train Acc: {acc / len(train_loader) * 100:>6.2f}% | Val Loss: '
                      f'{val_loss / len(train_loader):.2f} | Val Acc: '
                      f'{val_acc / len(train_loader) * 100:.2f}%')

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

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
    for node in graph.nodes:
        graph.nodes[node]['feature'] = str(node)
    return graph

def nx_to_torch_geometric(graph):
    """Convert a NetworkX graph to a PyTorch Geometric graph."""
    nodes = list(graph.nodes)
    node_mapping = {node: i for i, node in enumerate(nodes)}
    edges = [(node_mapping[src], node_mapping[dst]) for src, dst in graph.edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Create dummy features for each node
    features = torch.eye(len(nodes), dtype=torch.float)
    
    # Create random labels for each node (e.g., 2 classes)
    labels = torch.randint(0, 2, (len(nodes),), dtype=torch.long)

    # Create masks for training, validation, and testing
    train_mask = torch.zeros(len(nodes), dtype=torch.bool)
    val_mask = torch.zeros(len(nodes), dtype=torch.bool)
    test_mask = torch.zeros(len(nodes), dtype=torch.bool)

    # For simplicity, use all nodes for training
    train_mask[:] = True

    data = Data(x=features, edge_index=edge_index, y=labels, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return data

# Ensure file path is correct
file_path = 'practice_html/practice.html'  # If the file is in the current directory, otherwise provide a relative or absolute path

# Read HTML file content
with open(file_path, 'r', encoding='utf-8') as file:
    html_doc = file.read()

# Convert HTML to graph
graph = convert_html_to_graph(html_doc)
graph = generate_node_features(graph)

# Convert to PyTorch Geometric graph
data = nx_to_torch_geometric(graph)

# Create a GraphSAGE model
model = GraphSAGE(dim_in=data.num_node_features, dim_h=128, dim_out=2)  # Assuming 2 classes for output

# Define a DataLoader
loader = DataLoader([data], batch_size=1)

# Train the model
model.fit(data, loader, epochs=100)

# Get embedding vectors
model.eval()
embeddings = model(data.x, data.edge_index).detach().numpy()

# Print node embeddings
print(embeddings)

# Visualize the graph
plt.figure(figsize=(10, 10))
plt.title("GraphSAGE model Graph Visualization") 
pos = nx.spring_layout(graph, k=0.15)  # Increase k to reduce overlap
nx.draw(graph, pos, with_labels=True, node_size=700, node_color="plum", font_size=10, font_color="black")
plt.show()
