from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader,PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from sentence_transformers import SentenceTransformer
import uuid
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
import os
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

doc = Document(
    page_content="this is the main text content",
    metadata={
        "source": "example.txt",
        "pages": 1,
        "author": "suresh"
    }
)

print(doc)
## Loading a document from a text file splitting it into chunks and printing the chunks
loader = TextLoader("data/sample.txt")
document = loader.load()

splitter= RecursiveCharacterTextSplitter(chunk_size=100,
                                          chunk_overlap=20)
chunks = splitter.split_documents(document)

for chunk in chunks:
    print(chunk)
    print("--------------")

  ## Loading a document from a pdf file
pdf_loader = PyPDFLoader("data/rag_practice.pdf")
pdf_document = pdf_loader.load()
pdf_chunks = splitter.split_documents(pdf_document)


for chunk in pdf_chunks:
    print(chunk)
    print("--------------")  

    ### embeddings and vector store DB
    # Load embedding model
# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create chroma client
client = chromadb.Client()

# Create collection
collection = client.create_collection("my_collection")

# Documents
documents = [
    "AWS Bedrock provides AI services",
    "LangChain is useful for RAG",
    "Vector databases store embeddings"
]

# Generate embeddings
embeddings = model.encode(documents).tolist()

# Store in DB
collection.add(
    ids=["1", "2", "3"],
    documents=documents,
    embeddings=embeddings
)

print("Embeddings stored successfully")

## rag retrieval pipeline
query = "What does AWS Bedrock provide?"        
query_embedding = model.encode(query).tolist()
results = collection.query(
    query_embeddings=query_embedding,
    n_results=2
)
print("Retrieved documents:")
for doc in results['documents']:
    print(doc)