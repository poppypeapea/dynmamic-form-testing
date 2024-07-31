
Workflow:
Get a Website:
Convert the HTML to graphs.
Use GraphSAGE to generate embeddings for each node.
Calculate Similarity:
Compare each pair of nodes using cosine similarity.
Nodes: HTML nodes
Edges: Connections between every two nodes
Weights: Cosine similarity between node embeddings
Compare Two Versions of HTML Documents:
Normal Graph: Contains visual information (text, images).
Impaired Graph: Contains accessibility information (ARIA labels, alternative text).
Analyze and Improve:
Compare the two graphs to see the similarity.
If nodes are similar in the normal graph but not in the impaired graph, identify discrepancies.
Identify weak similarities in the impaired graph where strong similarities exist in the normal graph.




identify_impacted_nodes
generate_accessibility_suggestions
apply_suggestions_to_html