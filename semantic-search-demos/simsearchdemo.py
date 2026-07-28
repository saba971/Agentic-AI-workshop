from sklearn.metrics.pairwise import cosine_similarity
import ollama

text1 = "i like playing cricket"
text2 = "Artificial intelligence includes ML"

# text1 = "i like playing cricket"
# text2 = "i like playing cricket"


emb1 = ollama.embed(
    model='nomic-embed-text',
    input=text1
)['embeddings'][0]

emb2 = ollama.embed(
    model='nomic-embed-text',
    input=text2
)['embeddings'][0]

similarity = cosine_similarity([emb1], [emb2])

print(similarity)