import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Word embeddings for two words
word1_embedding = np.array([0.2, 0.5, 0.8, 0.3])
word2_embedding = np.array([0.4, 0.1, 0.9, 0.5])

print(word1_embedding)
print(word2_embedding)
print("==============")
# Reshape the arrays to match the expected input shape of cosine_similarity
word1_embedding = word1_embedding.reshape(1, -1)
word2_embedding = word2_embedding.reshape(1, -1)
print(word1_embedding)
print(word2_embedding)

# Calculate cosine similarity
similarity = cosine_similarity(word1_embedding, word2_embedding)[0][0]
print(similarity)