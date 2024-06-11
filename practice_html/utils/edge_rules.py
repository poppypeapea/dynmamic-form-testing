def add_parent_child_edges(graph, parent, soup_node):
    """Recursively add parent-child edges to the graph."""
    for child in soup_node.children:
        if child.name:  # Only consider tag elements
            graph.add_edge(str(parent), str(child))
            add_parent_child_edges(graph, child, child)

def add_label_input_edges(graph, soup):
    """Add edges between label and input elements."""
    for label in soup.find_all('label'):
        if 'for' in label.attrs:
            input_id = label['for']
            input_elem = soup.find(attrs={"id": input_id})
            if input_elem:
                graph.add_edge(str(label), str(input_elem))

def add_next_sibling_edges(graph, soup_node):
    """Add edges between next siblings."""
    for node in soup_node.descendants:
        if node.name and node.next_sibling and node.next_sibling.name:
            graph.add_edge(str(node), str(node.next_sibling))
