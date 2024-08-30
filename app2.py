import streamlit as st
from doc_load import load_doc
from chunks import chunk_text
from embeddings import create_embeddings
from datastore import store_vectors, load_vectors, save_faiss_index
from query import query_vector_store
from transformers import pipeline
import os
import numpy as np

@st.cache_resource
def get_faiss_index_and_chunks():
    if os.path.exists('faiss_index.index'):
        faiss_index = load_vectors('faiss_index.index')
        if os.path.exists('chunks.txt'):
            with open('chunks.txt', 'r', encoding='utf-8') as f:
                chunks = f.readlines()
        else:
            chunks = []  # Handle error or unexpected cases
    else:
        documents = load_doc('data')
        all_text = " ".join([doc.page_content for doc in documents])
        chunks = chunk_text(all_text, chunk_size=300)
        embeddings = create_embeddings(chunks)
        faiss_index = store_vectors(embeddings)

        # Save the FAISS index and chunks to disk
        save_faiss_index(faiss_index, "faiss_index.index")
        with open('chunks.txt', 'w', encoding='utf-8') as f:
            f.write("\n".join(chunks))

    return faiss_index, chunks

st.title("Veterinary Medicine RAG Application")

# Load or create FAISS index and chunks
faiss_index, chunks = get_faiss_index_and_chunks()

# Query interface
user_query = st.text_input("Ask a question:")
if user_query:
    query_embedding = create_embeddings([user_query])
    distances, indices = query_vector_store(faiss_index, query_embedding)

    # Flatten the indices array and ensure it is a list of integers
    indices = indices.flatten().astype(int).tolist()

    # Fetch the relevant text chunks based on the indices
    relevant_texts = [chunks[i] for i in indices]

    # Combine the relevant texts to use as context for the LLM
    context = " ".join(relevant_texts)

    # Initialize the LLM model (using a small model for the example)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # Generate the output using the context
    summary = summarizer(context, max_length=150, min_length=30, do_sample=False)

    st.write(f"### 1. Your query description:\n{user_query}")
    st.write(f"### 2. Summary of result:\n{summary[0]['summary_text']}")
    st.write(f"### 3. Conclusion:\nBased on the analysis, the most relevant information is presented.")
    st.write(f"### 4. Reference to Conclusion:\nThe conclusion is derived from the most relevant parts of the provided documents.")
    st.write(f"### 5. Disclaimer:\nLLM can be wrong, please correlate the results with your understandings and relevant texts.")


