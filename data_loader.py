import streamlit as st
from doc_load import load_doc
from chunks import chunk_text
from embeddings import create_embeddings
from datastore import store_vectors, load_vectors, save_faiss_index
from query import query_vector_store
import os
import json

@st.cache_resource
def get_faiss_index():
    documents = load_doc('data')
    all_text = " ".join([doc.page_content for doc in documents])
    chunks = chunk_text(all_text, chunk_size=300)
    embeddings, chunks = create_embeddings(chunks)  # Now we return both embeddings and chunks
    faiss_index, index_to_chunk_mapping = store_vectors(embeddings, chunks)  # Pass both embeddings and chunks

        # Save the FAISS index to disk
    save_faiss_index(faiss_index, "faiss_index_file.index")
    with open('index_to_chunk_mapping.json', 'w') as f:
            json.dump(index_to_chunk_mapping, f)
        
        # Return FAISS index and the mapping of indices to chunks
    return faiss_index, index_to_chunk_mapping


st.title("Veterinary Medicine RAG Application")

# Load or create FAISS index
faiss_index = get_faiss_index()

# Query interface
user_query = st.text_input("Ask a question:")
if user_query:
    query_embedding = create_embeddings([user_query])
    distances, indices = query_vector_store(faiss_index, query_embedding)
    st.write(f"Top results for your query: {indices}")
