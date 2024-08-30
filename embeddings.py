from sentence_transformers import SentenceTransformer

def create_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings, chunks  # Return both embeddings and chunks
