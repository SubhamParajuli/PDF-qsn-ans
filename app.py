import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

import streamlit as st

def create_text(pdf_files):
    text=""
    for pdf in pdf_files:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()




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

                #chunks to embedding

if __name__ == '__main__':
    main()
