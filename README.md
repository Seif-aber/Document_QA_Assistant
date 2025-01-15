# Document_QA_Assistant

Document_QA_Assistant is a document-based question-answering tool that uses RAG (Retrieval-Augmented Generation) with the LlamaIndex library for document retrieval and Gemini for embeddings and inference.

## Features

- Upload documents (e.g., `.pdf`, `.txt`) for processing.
- Ask questions about the content of the uploaded document.
- Efficient retrieval using LlamaIndex.
- High-quality embeddings and inference powered by Gemini.

## Demo

A live demo is available on Hugging Face Spaces:  
[Document_QA_Assistant Demo](https://huggingface.co/spaces/Seyfelislem/Document_QA_Assistant)

## How It Works

1. Upload a document.
2. The document is processed and indexed.
3. Ask a question about the document.
4. The system retrieves relevant content and generates an answer.

## Local Setup

To run the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Document_QA_Assistant.git
   cd Document_QA_Assistant
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the application:
   ```bash
   streamlit run app.py
