import streamlit as st
from doc_load import load_doc
from chunks import chunk_text
from embeddings import create_embeddings
from datastore import store_vectors, load_vectors
from query import query_vector_store

st.title("Veterinary Medicine RAG Application")

# File loading and processing logic
documents = load_doc('data')
# Combine text from all pages into one string
all_text = " ".join([doc.page_content for doc in documents])

# Chunk the text
chunks = chunk_text(all_text, chunk_size=300)
embeddings = create_embeddings(chunks)

# Store and load vectors
faiss_index = store_vectors(embeddings)

# Query interface
user_query = st.text_input("Ask a question:")
if user_query:
    query_embedding = create_embeddings([user_query])
    distances, indices = query_vector_store(faiss_index, query_embedding)
    st.write(f"Top results for your query: {indices}")
