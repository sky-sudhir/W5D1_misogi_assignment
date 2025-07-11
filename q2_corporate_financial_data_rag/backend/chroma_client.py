# chroma_client.py
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from config import settings
import structlog

logger = structlog.get_logger(__name__)


def get_vectorstore():
    """Get ChromaDB vectorstore instance"""
    try:
        # Use Ollama embeddings (you can switch to OpenAI if needed)
        embedding_fn = OllamaEmbeddings(model="nomic-embed-text")
        
        # Initialize ChromaDB client
        if settings.CHROMA_API_KEY and settings.CHROMA_TENANT and settings.CHROMA_DB:
            # Cloud client
            client = chromadb.CloudClient(
                api_key=settings.CHROMA_API_KEY,
                tenant=settings.CHROMA_TENANT,
                database=settings.CHROMA_DB
            )
            logger.info("Using ChromaDB Cloud client")
        else:
            # Local client
            client = chromadb.Client()
            logger.info("Using ChromaDB local client")
        
        # Create Chroma vectorstore
        vectorstore = Chroma(
            client=client,
            collection_name=settings.COLLECTION_NAME,
            embedding_function=embedding_fn
        )
        
        logger.info(f"ChromaDB vectorstore initialized with collection: {settings.COLLECTION_NAME}")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB vectorstore: {e}")
        raise


def get_collection_stats():
    """Get collection statistics"""
    try:
        vectorstore = get_vectorstore()
        
        # Get collection info
        collection = vectorstore._collection
        count = collection.count()
        
        stats = {
            "collection_name": settings.COLLECTION_NAME,
            "document_count": count,
            "embedding_model": "nomic-embed-text"
        }
        
        logger.info(f"Collection stats: {stats}")
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get collection stats: {e}")
        return {"error": str(e)}


def reset_collection():
    """Reset the collection (admin function)"""
    try:
        vectorstore = get_vectorstore()
        collection = vectorstore._collection
        
        # Delete all documents
        collection.delete()
        
        logger.info(f"Collection {settings.COLLECTION_NAME} reset successfully")
        return {"message": f"Collection {settings.COLLECTION_NAME} reset successfully"}
        
    except Exception as e:
        logger.error(f"Failed to reset collection: {e}")
        raise

