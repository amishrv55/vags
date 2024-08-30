import faiss
import numpy as np

import numpy as np

def store_vectors(embeddings, chunks):
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings.cpu().numpy())
    
    # Create the mapping from index to chunks
    index_to_chunk_mapping = {str(i): chunk for i, chunk in enumerate(chunks)}
    
    return index, index_to_chunk_mapping

def load_vectors(index_path):
    index = faiss.read_index(index_path)
    return index

# Save the FAISS index
def save_faiss_index(index, filepath):
    faiss.write_index(index, filepath)
