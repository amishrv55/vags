import streamlit as st
from doc_load import load_doc
from chunks import chunk_text
from embeddings import create_embeddings
from datastore import store_vectors, load_vectors, save_faiss_index
from query import query_vector_store
import openai
import os
import json
import numpy as np
from openai import OpenAI

# Set OpenAI API key

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],
)

@st.cache_resource
def load_faiss_and_mapping():
    faiss_index = load_vectors('faiss_index_file.index')
    with open('index_to_chunk_mapping.json', 'r') as f:
        index_to_chunk_mapping = json.load(f)
    return faiss_index, index_to_chunk_mapping

st.title("Veterinary Medicine RAG Application")

# Load FAISS index and mapping
faiss_index, index_to_chunk_mapping = load_faiss_and_mapping()

# Query interface
user_query = st.text_input("Ask a question:")
if user_query:
    query_embedding, _ = create_embeddings([user_query])
    distances, indices = query_vector_store(faiss_index, query_embedding)
    
    # Ensure indices are correctly converted to strings for lookup
    relevant_chunks = [index_to_chunk_mapping[str(i)] for i in indices[0]]

    # Combine the top chunks into context
    context = " ".join(relevant_chunks)
    
    # Create the message payload for the OpenAI chat completion
    messages = [
    {"role": "system", "content": f"User query,{user_query}:\n.You are a highly experienced Professor of Veterinary Medicine. Your role is to read the context, understand query and  provide detailed, step-by-step, and structured clinical advice to veterinarians, including drug names, dosages, dosing intervals, and contraindications. Ensure all advice is consistent and logically ordered."},
    {"role": "user", "content": f"Context: {context}\n\nRead the context and extarct relevant information from context to answer the user query.Based on the context provided, please provide the following:\n1. Your query description:\n2. A step-by-step analgesia protocol, including preoperative, intraoperative, and postoperative stages, with drug names, dosages, and dosing intervals:\n4. Reference to analgesia protocol."}
    
    ]

    # Call the OpenAI API to generate a response
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500,
        temperature=0,
    )
    
    # Parse the response from OpenAI
    llm_output = response.choices[0].message.content
    # Display the structured output
    st.subheader("LLM Response:")
    st.write(llm_output)


