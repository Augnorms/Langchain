import pandas as pd
import faiss
import numpy as np
import streamlit as st
import pickle
import time
import langchain
import os
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_community.vectorstores import FAISS

def chunk_summarizer(urls, userquery):
    # Initialize your model
    llm = OllamaLLM(model='tinyllama', temperature=0.3)
    
    # Load your data
    loaders = UnstructuredURLLoader(urls=urls)

    # Initialize your loaded data
    data = loaders.load()

    # Create chunks from your data to fit context window limit of your model
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Divide the loaded data into smaller, manageable chunks or documents
    docs = text_splitter.split_documents(data)

    # Use the HuggingFace model to create numerical representations of the docs
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create a vector index from the documents and their corresponding embeddings
    vector_index = FAISS.from_documents(docs, embeddings)
    
    # Saving the FAISS Index Locally
    vector_index.save_local("faiss_index")

    # Load the FAISS Index from storage
    index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Fetch relevant documents based on the input query
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=index.as_retriever())

    query = userquery

    langchain.debug = False

    result = chain({"question": query}, return_only_outputs=True)

    return result
