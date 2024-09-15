Veterinary RAG Model: A Novel Approach to Clinical Decision Support
Overview
The Veterinary RAG Model is a cutting-edge Retrieval-Augmented Generation (RAG) application designed to assist veterinary practitioners and students by providing detailed and specific clinical advice. By leveraging advanced machine learning techniques and a database of veterinary texts, our model aims to deliver accurate, relevant, and context-specific responses to clinical queries.

Features
Clinical Decision Support: Provides detailed clinical advice tailored to specific veterinary scenarios.
Scalable and Modular: Easily extendable with additional documents and use cases.
User-Friendly Interface: Designed with a simple interface for ease of use by veterinary professionals and students.
Customizable Outputs: Generates structured outputs with key information such as protocols, drug dosages, contraindications, and more.
OpenAI Integration: Uses OpenAI's GPT model to enhance the generation of accurate and contextually relevant advice.
Project Structure
```
├── app3.py                   # Main application file
├── chunks.py                 # Module for text chunking
├── datastore.py              # Handles storage and retrieval of embeddings and vectors
├── embeddings.py             # Functions to create embeddings
├── query.py                  # Handles queries to the vector store
├── doc_load.py               # Module to load and preprocess documents
├── requirements.txt          # List of required Python packages
├── README.md                 # Project documentation
├── index_to_chunk_mapping.json  # Mapping of FAISS index to text chunks
├── faiss_index.index         # FAISS index storing document embeddings
├── LICENSE                   # Project license

```
Installation
To run this project, you will need to set up a Python environment with the necessary dependencies.

Prerequisites
Python 3.8+
OpenAI API Key (for using GPT-based models)

Setup
Clone the repository:
'''
git clone https://github.com/amishrv55/vags.git
cd vags
'''
Usage
Once the application is running, you can input clinical queries into the text box provided. The model will retrieve relevant information from the document corpus and generate detailed clinical advice.

Example Query
Query: "Case 3: A 4-month old, healthy cat undergoing ovariohysterectomy or orchiectomy (spay-neuter program without opioid availability). Suggest an analgesia protocol with drugs and doses and dose intervals."

Output:

Query Description
Detailed Result with step-by-step protocol
Table with drug combinations, dosages, and observations
Conclusion and references
Contributing
We welcome contributions from the community! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature-name).
Make your changes.
Commit your changes (git commit -m 'Add some feature').
Push to the branch (git push origin feature-name).
Open a pull request.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
For questions or further information, please contact:

Amish Jain
Email: amishjain2025@u.northwestern.edu
Project: Veterinary RAG Model
