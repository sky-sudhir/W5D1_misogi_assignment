# chroma_client.py
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

def get_vectorstore():
    embedding_fn = OllamaEmbeddings(model="nomic-embed-text")

    client = chromadb.CloudClient(
  api_key=os.getenv('CHROMA_API_KEY'),
  tenant=os.getenv('CHROMA_TENANT'),
  database=os.getenv('CHROMA_DB')
)
    return Chroma(
        client=client,
        collection_name="drugbank_chunks",
        embedding_function=embedding_fn
    )

