import os
import faiss
import pickle
from dotenv import load_dotenv
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to load text data (for now, simple)
def load_text_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Function to split text into chunks
def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    return text_splitter.split_text(text)

# Function to create FAISS vector store
def create_faiss_index(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore

# Save FAISS index locally
# Save FAISS index properly
def save_faiss_index(vectorstore, path="faiss_index"):
    vectorstore.save_local(path)

if __name__ == "__main__":
    # Example file (replace with your own later)
    file_path = "sample.txt"

    # Step 1: Load document
    text = load_text_data(file_path)

    # Step 2: Split into chunks
    chunks = split_text(text)

    # Step 3: Embed and store in FAISS
    vectorstore = create_faiss_index(chunks)

    # Step 4: Save FAISS index
    save_faiss_index(vectorstore)

    print("✅ Vectorstore created and saved successfully!")