from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
import os

# initializing path for documents in root folder
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "documents", "lord_of_the_rings.txt")


def retrieval_augmented_gen():
    model = OllamaLLM(model="phi3:latest")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
                f"The file {file_path} does not exist"
        )
            
    loader = TextLoader(file_path)
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

    docs = text_splitter.split_documents(document)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_index = FAISS.from_documents(docs, embeddings)

    # Saving the FAISS Index Locally
    save_to_db = "vector_db"
    if not os.path.exists(save_to_db):
        vector_index.save_local(save_to_db)
    else:
        # Load the FAISS Index from storage
        index = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)

        # Configure the retriever to fetch a specific number of documents
        retriever = index.as_retriever(search_kwargs={"k": 3}) 

        # Fetch relevant documents based on the input query
        chain = RetrievalQAWithSourcesChain.from_llm(llm=model, retriever=retriever, return_source_documents=True)

        query = "Where did Gandalf meet Frodo?"

        result = chain.invoke({"question": query})


        print(result)

    return