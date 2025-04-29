import os
import streamlit as st
import fitz  # PyMuPDF for PDF reading
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------- CONSTANTS ---------- #
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
LLM_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"

# ---------- LOAD ENV & API KEY ---------- #
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("\ud83d\udeab Missing OpenAI API key. Please set it in your `.env` or Streamlit Secrets.")
    st.stop()

# ---------- INITIALIZE LLM ---------- #
llm = ChatOpenAI(openai_api_key=api_key, model=LLM_MODEL)

# ---------- STREAMLIT PAGE CONFIG ---------- #
st.set_page_config(page_title="\ud83e\uddb0 Brainwave AI", page_icon="\ud83e\uddb0")
st.title("\ud83e\uddb0 Brainwave AI: Chat with Your Knowledge")

st.caption("Upload a document (.txt or .pdf) and chat with it using AI!")
st.divider()

# ---------- PREMIUM CSS ---------- #
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f0f0f5 0%, #e8eaed 100%);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
[data-testid="stChatMessage"] div[role="button"] {
    border-radius: 12px; padding: 8px 12px; margin-bottom: 6px;
}
[data-testid="stChatMessage"][aria-label="assistant"] div[role="button"] {
    background-color: #e0f7fa;
}
[data-testid="stChatMessage"][aria-label="user"] div[role="button"] {
    background-color: #f0f0f5;
}
</style>
""", unsafe_allow_html=True)

# ---------- FUNCTION DEFINITIONS ---------- #
def load_document(uploaded_file):
    """Load and extract text from txt/pdf file."""
    try:
        if uploaded_file.name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8")
        elif uploaded_file.name.endswith(".pdf"):
            pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = "".join([page.get_text() for page in pdf_doc])
            if not text.strip():
                st.error("⚠️ Uploaded PDF has no readable text.")
                st.stop()
            return text
        else:
            st.error("❌ Unsupported file type. Upload a .txt or .pdf.")
            st.stop()
    except Exception as e:
        st.error(f"⚡ Error reading file: {e}")
        st.stop()

def build_vectorstore(text):
    """Split text, embed, and create FAISS vectorstore."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(text)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.from_texts(chunks, embedding=embeddings)

# ---------- SESSION STATE ---------- #
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# ---------- FILE UPLOADER ---------- #
uploaded_file = st.file_uploader("Upload a document to start chatting:", type=["txt", "pdf"])

if uploaded_file:
    st.success(f"✅ Uploaded: {uploaded_file.name}")

    with st.spinner("🔍 Reading your document..."):
        file_text = load_document(uploaded_file)

    with st.spinner("🔗 Creating knowledge base..."):
        try:
            vectorstore = build_vectorstore(file_text)
            st.session_state.qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
        except Exception as e:
            st.error(f"⚡ Error building vectorstore: {e}")
            st.stop()

    # Reset chat messages
    st.session_state.messages.clear()

st.divider()

# ---------- CHAT INTERFACE ---------- #
if st.session_state.qa_chain:

    # Show previous conversation
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    query = st.chat_input("Ask anything about your document...")

    if query:
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("\ud83e\uddec Thinking..."):
            response = st.session_state.qa_chain.run(query)

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# ---------- FOOTER ---------- #
st.divider()
st.caption("Made with ❤️ using OpenAI and LangChain")

