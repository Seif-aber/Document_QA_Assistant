import streamlit as st
import os
from utils import load_data, get_gemini_embedding


def process_document(doc, question):
    """Process document and return response to question."""
    temp_path = os.path.join("data", doc.name)
    try:
        with open(temp_path, "wb") as f:
            f.write(doc.getbuffer())
        documents = load_data("data")
        query_engine = get_gemini_embedding(documents)
        return query_engine.query(question)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def main():
    st.set_page_config(page_title="Document Q&A Assistant")
    st.title("Smart Document Question-Answering")

    # Create data directory if not exists
    os.makedirs("data", exist_ok=True)

    doc = st.file_uploader(
        "Upload your document (PDF, CSV, or TXT)", type=["pdf", "csv", "txt"]
    )

    question = st.text_input(
        "What would you like to know about your document?",
        placeholder="Enter your question here...",
    )

    if st.button("Get Answer"):
        if not doc:
            st.error("Please upload a document first.")
            return
        if not question:
            st.error("Please enter a question.")
            return

        with st.spinner("Analyzing your document..."):
            response = process_document(doc, question)
            st.write(response.response)


if __name__ == "__main__":
    main()
