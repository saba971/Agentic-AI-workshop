from ollama import embeddings

response = embeddings(
    model='nomic-embed-text',
    prompt='What is Artificial Intelligence?'
)

print(response['embedding'])
print(response['embedding'][:10])  # first 10 values