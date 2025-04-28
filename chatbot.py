import os
import pickle
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load FAISS vectorstore properly
def load_vectorstore(path="faiss_index"):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = FAISS.load_local(
        folder_path=path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True  # <=== ADD THIS
    )
    return vectorstore

# Build the RetrievalQA chain
def build_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

if __name__ == "__main__":
    # Step 1: Load Vectorstore
    vectorstore = load_vectorstore()

    # Step 2: Build Retrieval Chain
    qa_chain = build_chain(vectorstore)

    # Step 3: Chat loop
    print("🧠 Chatbot ready! (type 'exit' to quit)")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        response = qa_chain.run(query)
        print("Bot:", response)