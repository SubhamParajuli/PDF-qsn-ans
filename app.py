import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st
import torch

def create_text(pdf_files):
    text=""
    for pdf in pdf_files:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_chunks(text):
    text_splitter=CharacterTextSplitter(
        seperator='\n',
        chunk_size=1000,
        chunk_overlap=200
    )


def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon="📚")
    
    st.subheader("OK Google:")
    st.header("Chat with multiple PDFs 📚")
    
    st.text_input("Ask a question:")

    with st.sidebar:
        st.subheader("Your documents:")
        pdf_files=st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        if(st.button("Process")) :

            with st.spinner():
                #get pdf files
                raw_text=create_text(pdf_files)

                #text-> chunks
                chunks=get_chunks(raw_text)

                #chunks to embedding

if __name__ == '__main__':
    main()
