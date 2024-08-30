import streamlit as st
from doc_load import load_doc
from chunks import chunk_text
from embeddings import create_embeddings
from datastore import store_vectors, load_vectors, save_faiss_index
from query import query_vector_store
import os

@st.cache_resource
def get_faiss_index():
    if os.path.exists('faiss_index.index'):
        return load_vectors('faiss_index.index')
    else:
        documents = load_doc('data')
        all_text = " ".join([doc.page_content for doc in documents])
        chunks = chunk_text(all_text, chunk_size=300)
        embeddings = create_embeddings(chunks)
        faiss_index = store_vectors(embeddings)

        # Save the FAISS index to disk
        save_faiss_index(faiss_index, "faiss_index_file.index")
        return faiss_index

st.title("Veterinary Medicine RAG Application")

# Load or create FAISS index
faiss_index = get_faiss_index()

# Query interface
user_query = st.text_input("Ask a question:")
if user_query:
    query_embedding = create_embeddings([user_query])
    distances, indices = query_vector_store(faiss_index, query_embedding)
    st.write(f"Top results for your query: {indices}")
