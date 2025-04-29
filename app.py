import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF for PDF reading

# Load and check OpenAI API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("🚫 Missing OpenAI API key. Please set it in your `.env` or in Streamlit Secrets.")
    st.stop()

# Initialize LLM
llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo")

# Streamlit Page Config & Title
st.set_page_config(page_title="🧠 Brainwave AI: Chat with Your Knowledge", page_icon="🧠")
st.title("🧠 Brainwave AI: Chat with Your Knowledge")

# ——— PREMIUM CSS INJECTION ———
st.markdown(
    """
    <style>
    /* Global background & font */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f0f0f5 0%, #e8eaed 100%);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    /* File uploader styling */
    input[type="file"] {
        border: 2px dashed #bbb !important;
        border-radius: 8px !important;
        padding: 20px !important;
        background-color: #ffffff !important;
        transition: border-color 0.3s ease;
    }
    input[type="file"]:hover {
        border-color: #888 !important;
    }
    /* Chat input styling */
    textarea, input[type="text"] {
        border: none !important;
        border-bottom: 2px solid #ccc !important;
        background-color: transparent !important;
        color: #1c1c1e !important;
        font-size: 17px !important;
        padding: 10px 5px !important;
        transition: border-color 0.3s !important;
    }
    textarea:focus, input[type="text"]:focus {
        border-bottom: 2px solid #007aff !important;
        outline: none !important;
    }
    /* Spinner text */
    [data-testid="stSpinner"] {
        font-style: italic;
        color: #555;
    }
    /* Chat bubbles */
    [data-testid="stChatMessage"] div[role="button"] {
        border-radius: 12px !important;
        padding: 8px 12px !important;
        margin-bottom: 6px !important;
    }
    /* Assistant bubble color */
    [data-testid="stChatMessage"][aria-label="assistant"] div[role="button"] {
        background-color: #e0f7fa !important;
    }
    /* User bubble color */
    [data-testid="stChatMessage"][aria-label="user"] div[role="button"] {
        background-color: #f0f0f5 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to read uploaded file
def read_file(uploaded_file):
    if uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".pdf"):
        pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in pdf_doc:
            text += page.get_text()
        if not text.strip():
            st.error("⚠️ Your uploaded PDF has no readable text. Please upload a different file.")
            st.stop()
        return text
    else:
        st.error("❌ Unsupported file type. Please upload a .txt or .pdf file.")
        st.stop()

# Function to create FAISS vectorstore
def create_vectorstore(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore

# File Uploader
uploaded_file = st.file_uploader("Upload a .txt or .pdf to chat with:", type=["txt", "pdf"])

if uploaded_file:
    with st.spinner("📄 Processing your document..."):
        file_text = read_file(uploaded_file)
    vectorstore = create_vectorstore(file_text)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    # Initialize session messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # Chat input
    query = st.chat_input("Ask something about your uploaded document...")

    if query:
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("🧠 Brainwave is thinking..."):
            response = qa_chain.run(query)

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


