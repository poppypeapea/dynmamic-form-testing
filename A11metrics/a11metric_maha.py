import os
import networkx as nx
from bs4 import BeautifulSoup
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import openai
from graphsage import GraphSAGE
from dotenv import load_dotenv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances
import numpy as np

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

def get_ada_embedding(text):
    return openai.Embedding.create(input=[text], model='text-embedding-ada-002').data[0].embedding

def convert_html_to_graph(html_doc):
    soup = BeautifulSoup(html_doc, 'html.parser')
    graph = nx.Graph()

    for element in soup.find_all():
        node_id = f"{element.name}_{element.get('id', '')}_{element.get('class', '')}"
        graph.add_node(node_id, tag_name=element.name, attrs=element.attrs)
    return graph

def generate_node_features(graph):
    ada_embedding_size = 1536
    zero_vector = [0] * ada_embedding_size

    for node in graph.nodes:
        attrs = graph.nodes[node]['attrs']
        text_content = " ".join(attrs.get('id', '') + " " + " ".join(attrs.get('class', '')))
        feature = get_ada_embedding(text_content) if text_content else zero_vector
        graph.nodes[node]['feature'] = feature
    return graph

def nx_to_torch_geometric(graph):
    nodes = list(graph.nodes)
    node_mapping = {node: i for i, node in enumerate(nodes)}
    
    edges = [(node_mapping[src], node_mapping[dst]) for src in nodes for dst in nodes if src != dst]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    features = torch.tensor([graph.nodes[node]['feature'] for node in nodes], dtype=torch.float)
    
    num_nodes = len(nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_split = int(0.8 * num_nodes)
    val_split = int(0.9 * num_nodes)

    train_mask[:train_split] = True
    val_mask[train_split:val_split] = True
    test_mask[val_split:] = True

    labels = torch.randint(0, 2, (num_nodes,), dtype=torch.long)

    data = Data(x=features, edge_index=edge_index, y=labels, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return data

def add_distance_weights(graph, embeddings, metric='manhattan'):
    node_list = list(graph.nodes)
    if metric == 'cosine':
        dist_matrix = cosine_similarity(embeddings)
    elif metric == 'manhattan':
        dist_matrix = manhattan_distances(embeddings)
    else:
        raise ValueError("Unsupported metric")

    for i, src in enumerate(node_list):
        for j, dst in enumerate(node_list):
            if i != j:
                distance = dist_matrix[i, j]
                graph.add_edge(src, dst, weight=distance)
    return graph

def calculate_similarities(data, model):
    loader = DataLoader([data], batch_size=1)
    model.eval()
    initial_embeddings = model(data.x, data.edge_index).detach().numpy()
    model.train()
    model.fit(data, loader, epochs=100)
    model.eval()
    trained_embeddings = model(data.x, data.edge_index).detach().numpy()
    return initial_embeddings, trained_embeddings

def compare_graphs(normal_graph, impaired_graph, normal_embeddings, impaired_embeddings, metric='manhattan'):
    if metric == 'cosine':
        normal_dist_matrix = cosine_similarity(normal_embeddings)
        impaired_dist_matrix = cosine_similarity(impaired_embeddings)
    elif metric == 'manhattan':
        normal_dist_matrix = manhattan_distances(normal_embeddings)
        impaired_dist_matrix = manhattan_distances(impaired_embeddings)
    else:
        raise ValueError("Unsupported metric")

    discrepancies = []
    
    for node in normal_graph.nodes:
        if node in impaired_graph.nodes:
            i = list(normal_graph.nodes).index(node)
            j = list(impaired_graph.nodes).index(node)
            normal_dist = normal_dist_matrix[i]
            impaired_dist = impaired_dist_matrix[j]
            print(f"Node {node}: Normal Mean Distance = {np.mean(normal_dist)}, Impaired Mean Distance = {np.mean(impaired_dist)}")  # 调试输出
            if np.mean(normal_dist) > 0.01 and np.mean(impaired_dist) < 0.01:
                discrepancies.append((node, np.mean(normal_dist), np.mean(impaired_dist)))
    
    return discrepancies

def visualize_embeddings(embeddings, title):
    tsne_model = TSNE(n_components=2, perplexity=5, random_state=42)
    embeddings_2d = tsne_model.fit_transform(embeddings)
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=20)
    plt.show()

# Read HTML file content
normal_file_path = os.path.join(os.path.dirname(__file__), 'normal.html')
impaired_file_path = os.path.join(os.path.dirname(__file__), 'impaired.html')

with open(normal_file_path, 'r', encoding='utf-8') as file:
    normal_html_doc = file.read()

with open(impaired_file_path, 'r', encoding='utf-8') as file:
    impaired_html_doc = file.read()

# Convert HTML to graph
normal_graph = convert_html_to_graph(normal_html_doc)
impaired_graph = convert_html_to_graph(impaired_html_doc)

# Generate node features
normal_graph = generate_node_features(normal_graph)
impaired_graph = generate_node_features(impaired_graph)

# Convert to PyTorch Geometric graph
normal_data = nx_to_torch_geometric(normal_graph)
impaired_data = nx_to_torch_geometric(impaired_graph)

# Check if data contains the correct masks and labels
print(f"Train Mask (Normal): {normal_data.train_mask}")
print(f"Validation Mask (Normal): {normal_data.val_mask}")
print(f"Test Mask (Normal): {normal_data.test_mask}")
print(f"Labels (Normal): {normal_data.y}")

print(f"Train Mask (Impaired): {impaired_data.train_mask}")
print(f"Validation Mask (Impaired): {impaired_data.val_mask}")
print(f"Test Mask (Impaired): {impaired_data.test_mask}")
print(f"Labels (Impaired): {impaired_data.y}")

# Calculate similarities
initial_normal_embeddings, trained_normal_embeddings = calculate_similarities(normal_data, GraphSAGE(dim_in=normal_data.num_node_features, dim_h=128, dim_out=128))
initial_impaired_embeddings, trained_impaired_embeddings = calculate_similarities(impaired_data, GraphSAGE(dim_in=impaired_data.num_node_features, dim_h=128, dim_out=128))

# Print embeddings before and after training
print("Initial Normal Embeddings:\n", initial_normal_embeddings)
print("Trained Normal Embeddings:\n", trained_normal_embeddings)
print("Initial Impaired Embeddings:\n", initial_impaired_embeddings)
print("Trained Impaired Embeddings:\n", trained_impaired_embeddings)

# Add distance weights
normal_graph = add_distance_weights(normal_graph, trained_normal_embeddings, metric='manhattan')
impaired_graph = add_distance_weights(impaired_graph, trained_impaired_embeddings, metric='manhattan')

# Print similarity matrices for debugging
normal_dist_matrix = manhattan_distances(trained_normal_embeddings)
impaired_dist_matrix = manhattan_distances(trained_impaired_embeddings)
print("Normal Distance Matrix:\n", normal_dist_matrix)
print("Impaired Distance Matrix:\n", impaired_dist_matrix)

# Compare graphs
discrepancies = compare_graphs(normal_graph, impaired_graph, trained_normal_embeddings, trained_impaired_embeddings, metric='manhattan')

# Print discrepancies
print("Discrepancies between normal and impaired graphs (Manhattan):")
for node, normal_dist, impaired_dist in discrepancies:
    print(f"Node {node}: Normal distance = {normal_dist:.2f}, Impaired distance = {impaired_dist:.2f}")

# Visualize embeddings
visualize_embeddings(initial_normal_embeddings, "Initial Normal Embeddings")
visualize_embeddings(trained_normal_embeddings, "Trained Normal Embeddings")
visualize_embeddings(initial_impaired_embeddings, "Initial Impaired Embeddings")
visualize_embeddings(trained_impaired_embeddings, "Trained Impaired Embeddings")
