import os
import networkx as nx
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import openai
from utils.graphsage import GraphSAGE
from utils.edge_rules import add_parent_child_edges, add_label_input_edges, add_next_sibling_edges  
from dotenv import load_dotenv

load_dotenv() 

api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

def get_ada_embedding(text):
    return openai.Embedding.create(input=[text], model='text-embedding-ada-002').data[0].embedding

def get_detailed_description(element, context):
    prompt = f"Given the context '{context}', provide a detailed description for the following HTML element in 50 words based on the specific html: {element}. Consider the structure, purpose, and user interactions."
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']

# HTML to Graph Conversion Functions
def convert_html_to_graph(html_doc):
    """Convert an HTML document into a graph."""
    soup = BeautifulSoup(html_doc, 'html.parser')
    graph = nx.DiGraph()

    root = soup.find()
    graph.add_node(str(root))
    add_parent_child_edges(graph, root, root)
    add_label_input_edges(graph, soup)
    add_next_sibling_edges(graph, soup)
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
    ada_embedding_size = 1536  # Size of the ADA embedding
    zero_vector = [0] * ada_embedding_size

    for node in graph.nodes:
        soup_node = BeautifulSoup(node, 'html.parser').find()
        tag = soup_node.name if soup_node else 'other'
        if soup_node and soup_node.string:  # features of the embeddings
            graph.nodes[node]['feature'] = get_ada_embedding(soup_node.string)
        else:
            graph.nodes[node]['feature'] = zero_vector
        graph.nodes[node]['label'] = labels.get(tag, labels['other'])
    return graph

def nx_to_torch_geometric(graph):
    """Convert a NetworkX graph to a PyTorch Geometric graph."""
    nodes = list(graph.nodes)
    node_mapping = {node: i for i, node in enumerate(nodes)}
    edges = [(node_mapping[src], node_mapping[dst]) for src, dst in graph.edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Extract features from nodes
    features = torch.tensor([graph.nodes[node]['feature'] for node in nodes], dtype=torch.float)
    
    # Use the labels generated in generate_node_features
    labels = torch.tensor([graph.nodes[node]['label'] for node in nodes], dtype=torch.long)

    # Create masks for training, validation, and testing
    train_mask = torch.zeros(len(nodes), dtype=torch.bool)
    val_mask = torch.zeros(len(nodes), dtype=torch.bool)
    test_mask = torch.zeros(len(nodes), dtype=torch.bool)

    # For simplicity, use all nodes for training
    train_mask[:] = True

    data = Data(x=features, edge_index=edge_index, y=labels, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return data

# Ensure file path is correct
file_path = os.path.join(os.path.dirname(__file__), 'practice.html')

# Read HTML file content
with open(file_path, 'r', encoding='utf-8') as file:
    html_doc = file.read()

# Convert HTML to graph
graph = convert_html_to_graph(html_doc)
graph = generate_node_features(graph)

# Convert to PyTorch Geometric graph
data = nx_to_torch_geometric(graph)

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

# Find the "see more" element and its context
context_mapping = {}
for idx, node in enumerate(graph.nodes):
    soup_node = BeautifulSoup(node, 'html.parser').find()
    if soup_node:
        element_html = str(soup_node)
        context_mapping[element_html] = embeddings[idx]

# Calculate cosine similarity between each node and "see more" node
see_more_embedding = None
see_more_idx = None

for idx, node in enumerate(graph.nodes):
    soup_node = BeautifulSoup(node, 'html.parser').find()
    if soup_node and 'see more' in soup_node.get_text().lower():
        see_more_embedding = embeddings[idx]
        see_more_idx = idx
        break

if see_more_embedding is not None:
    similarities = {}
    for idx, embedding in enumerate(embeddings):
        similarity = cosine_similarity([embedding], [see_more_embedding])[0][0]
        similarities[idx] = similarity

    # Print cosine similarities
    for idx, similarity in similarities.items():
        print(f"Node {idx, node} - Cosine Similarity: {similarity:.4f}")

'''
# Generate detailed description for the "see more" element with related products context
if see_more_idx is not None:
    soup_node = BeautifulSoup(list(graph.nodes)[see_more_idx], 'html.parser').find()
    if soup_node:
        # Get related products context
        related_products = []
        for sibling in soup_node.parent.find_all_previous():
            if sibling.name == 'div' and sibling.get('class') == ['item']:
                related_products.append(sibling.get_text())

        context = " ".join(related_products)
        description = get_detailed_description('see more', context)
        print(f"Element: {'see more'}\nDescription: {description}\n")
'''