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

# Set your OpenAI API key

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

@st.cache_resource
def load_faiss_and_mapping():
    faiss_index = load_vectors('faiss_index.index')
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
    {"role": "system", "content": "You are a Professor specialized in Veterinary Medicine. Your job is to provide detailed and specific clinical protocols to veterinarians."},
    {"role": "user", "content": f"Context: {context}\n\nBased on the above context, please provide the following:\n1. Your query description:\n2. Detailed result:\n3. Output in form of table or Dataframe if asked in query:\n4. Conclusion:\n5. Reference to Conclusion:\n6. Disclaimer: 'LLM can be wrong, please correlate the results with your understandings and relevant texts.'"}
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


