from importlib import import_module

import streamlit as st

st.title("PDF Chunk Search")

books = {
    "Python": "rag_1",
    "Machine Learning": "rag_2",
    "Data Visualization": "rag_3",
}

book = st.sidebar.selectbox("Choose a PDF", list(books))


@st.cache_resource
def load_retriever(module_name):
    return import_module(module_name).get_relevant_info


question = st.chat_input(f"Ask a question about {book}")

if question and question.strip():
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the PDF..."):
            retrieve = load_retriever(books[book])
            chunks = retrieve(question.strip(), top_k=3)

        if not chunks:
            st.info("No chunks were returned.")

        for index, chunk in enumerate(chunks, start=1):
            st.subheader(f"Result {index}")
            st.text(chunk)