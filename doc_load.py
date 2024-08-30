from langchain_community.document_loaders import PyMuPDFLoader
import os

def load_doc(file_path):
    documents = []
    for filename in os.listdir(file_path):
        path = os.path.join(file_path,filename)
        loader = PyMuPDFLoader(path)
        documents.extend(loader.load())
    return documents
