import os
import openai
from bs4 import BeautifulSoup, Comment
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

def identify_impacted_nodes(discrepancies, threshold=5.0):
    impacted_nodes = [node for node, normal_dist, impaired_dist in discrepancies if impaired_dist - normal_dist > threshold]
    return impacted_nodes

def generate_accessibility_suggestions(node_name):
    prompt = f"The HTML node <{node_name}> has shown significant changes in its attributes or position, affecting accessibility. Suggest improvements to enhance its accessibility."
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    suggestions = response.choices[0].message['content'].strip()
    return suggestions

def apply_suggestions_to_html(html_doc, suggestions):
    soup = BeautifulSoup(html_doc, 'html.parser')
    for node in soup.find_all():
        node_id = f"{node.name}_{node.get('id', '')}_{node.get('class', '')}"
        if node_id in suggestions:
            comment = Comment(f"Accessibility Suggestion: {suggestions[node_id]}")
            node.insert_after(comment)
    return str(soup)

# Example discrepancies output
discrepancies = [
    ("img__", 2.18, 2.73),
    ("footer__", 2.18, 2.73),
    ("contact_", 19.50, 26.13)
]

# Identify impacted nodes
impacted_nodes = identify_impacted_nodes(discrepancies)

# Generate suggestions
suggestions = {}
for node in impacted_nodes:
    suggestion = generate_accessibility_suggestions(node)
    suggestions[node] = suggestion

# Print suggestions
for node, suggestion in suggestions.items():
    print(f"Node: {node}\nSuggestion: {suggestion}\n")

# More problematic impaired HTML document
impaired_html_doc = """
<!DOCTYPE html>
<html>
<head>
    <title>Sample Impaired Document</title>
    <style>
        body { font-size: small; }
        h1 { color: red; }
    </style>
</head>
<body>
    <header>
        <h1>Welcome</h1>
    </header>
    <main id="main_content">
        <section>
            <h2>About</h2>
            <p>This is a sample website with minimal information.</p>
            <img src="image.jpg">
        </section>
    </main>
    <footer>
        <address id="contact">
            <p>Contact us at info@example.com</p>
        </address>
    </footer>
</body>
</html>
"""

# Apply the suggestions to the HTML
enhanced_html_doc = apply_suggestions_to_html(impaired_html_doc, suggestions)

# Print the enhanced HTML
print(enhanced_html_doc)
