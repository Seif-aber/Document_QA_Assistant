from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
import logging
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_data(data_path: str) -> list[str]:
    """
    Load documents from a directory.
    
    Args:
        data_path (str): Path to the directory containing documents
    
    Returns:
        list[str]: List of loaded documents or False if loading fails
    """
    try:
        logger.info(f"Loading documents from {data_path}")
        loader = SimpleDirectoryReader(data_path)
        documents = loader.load_data()
        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        return False

def get_gemini_embedding(documents: str):
    """
    Create a query engine using Gemini embeddings.
    
    Args:
        documents (str): Documents to process
    
    Returns:
        QueryEngine: Configured query engine or False if setup fails
    """
    try:
        logger.info("Initializing Gemini embedding model and LLM")
        gemini_embedding_model = GeminiEmbedding(model_name="models/embedding-001")
        llm = Gemini(model="models/gemini-1.5-flash", api_key=GEMINI_API_KEY)

        # Configure global settings
        Settings.llm = llm
        Settings.embed_model = gemini_embedding_model
        Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=20)

        logger.info("Creating vector store index")
        index = VectorStoreIndex.from_documents(
            documents=documents, 
            embed_model=gemini_embedding_model
        )
        
        logger.info("Creating query engine")
        return index.as_query_engine()
    except Exception as e:
        logger.error(f"Failed to setup Gemini embedding: {str(e)}")
        return False