import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
import os

# Load Gemini API key
load_dotenv()

def get_gemini_llm():
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    return GoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

# Extract text from PDFs
def create_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

# Chunk the text
def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# Embed and store chunks in FAISS
def get_vectorstore(txt_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=txt_chunks, embedding=embeddings)
    return vectorstore

# Streamlit UI
def main():
    st.set_page_config(page_title="Chat with PDFs + Gemini + Web Context", page_icon="📚")
    st.header("Pdf Questionare:🗃️")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask a question:")

    with st.sidebar:
        st.subheader("Upload your documents:")
        pdf_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

        if st.button("Process") and pdf_files:
            with st.spinner("Reading and indexing PDFs..."):
                raw_text = create_text(pdf_files)
                chunks = get_chunks(raw_text)
                vector_store = get_vectorstore(chunks)
                st.session_state.vector_store = vector_store
                st.session_state.llm = get_gemini_llm()
                st.success("✅ Vector store & Gemini model initialized!")

    # Handle Q&A
    if user_question and "vector_store" in st.session_state and "llm" in st.session_state:
        with st.spinner("Thinking deeply with Gemini..."):
            # Step 1: Similarity search from FAISS
            docs = st.session_state.vector_store.similarity_search(user_question, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])

            # Step 2: Custom prompt for Gemini
            prompt = f"""
You are an intelligent research assistant. Answer the following question in detail and clarity. Use the context below if useful, but feel free to include extra information from your general knowledge and web-scale understanding.

Context from PDF documents:
{context}

Question: {user_question}

Answer in a detailed, helpful, and well-structured format:
            """

            # Step 3: Use Gemini LLM directly
            response = st.session_state.llm.invoke(prompt)

        # Save to chat history
        st.session_state.chat_history.append(("🧑 You", user_question))
        st.session_state.chat_history.append(("🤖 Gemini", response))

    # Display chat history
    for role, message in st.session_state.chat_history:
        st.markdown(f"**{role}:** {message}")

if __name__ == '__main__':
    main()
