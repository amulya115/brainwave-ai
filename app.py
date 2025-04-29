import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz
load_dotenv()

# Check and load OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("🚫 Missing OpenAI API key. Please set it in your `.env` or Streamlit secrets.")
    st.stop()

llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo")

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

def create_vectorstore(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore

st.set_page_config(page_title="🧠 Brainwave AI: Chat with Your Knowledge", page_icon="🧠")
st.title("🧠 Brainwave AI: Chat with Your Knowledge")
# True Luxury Style Input
st.markdown(
    """
    <style>
    textarea, input {
        border: none;
        border-bottom: 2px solid #e0e0e0;
        background-color: transparent;
        color: #1c1c1e;
        font-size: 17px;
        padding: 10px 5px;
        transition: border-color 0.3s;
    }
    textarea:focus, input:focus {
        border-bottom: 2px solid #007aff; /* Light soft blue underline */
        outline: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Upload a .txt or .pdf file to chat with:", type=["txt", "pdf"])

if uploaded_file:
    with st.spinner("Processing your document..."):
        file_text = read_file(uploaded_file)
        if file_text:
            vectorstore = create_vectorstore(file_text)
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

            
            if "messages" not in st.session_state:
                st.session_state.messages = []

            
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.chat_message("user").markdown(msg["content"])
                else:
                    st.chat_message("assistant").markdown(msg["content"])

            
            query = st.chat_input("Ask something about your uploaded document...")

            if query:
                st.chat_message("user").markdown(query)
                st.session_state.messages.append({"role": "user", "content": query})

                with st.spinner("🧠 Brainwave is thinking..."):
                    response = qa_chain.run(query)
                st.chat_message("assistant").markdown(f"✨ {response}")
                st.session_state.messages.append({"role": "assistant", "content": response})

        else:
            st.error("Unsupported file type! Please upload a .txt or .pdf file.")

