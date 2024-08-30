def query_vector_store(index, query_embedding):
    distances, indices = index.search(query_embedding, k=10)  # Adjust `k` for the number of results
    return distances, indices
