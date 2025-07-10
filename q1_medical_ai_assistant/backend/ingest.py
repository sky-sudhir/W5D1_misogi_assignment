
import pandas as pd
import fitz  # PyMuPDF
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document




# --- CSV Ingestion (DrugBank, RxNorm etc.) ---
def load_drugbank_csv(file_path: str):
    df = pd.read_csv(file_path)
    docs = []

    for _, row in df.iterrows():
        content = "\n".join([f"{col}: {row[col]}" for col in df.columns if pd.notnull(row[col])])
        docs.append(Document(page_content=content))

    return docs


# --- PDF Ingestion (WHO, NICE, CDC guidelines etc.) ---
def load_pdf(file_path: str):
    doc = fitz.open(file_path)
    
    full_text = ""

    for page in doc:
        full_text += page.get_text()

    return [Document(page_content=full_text)]


# --- Dispatcher Function for Ingestion ---
def ingest_file(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        raw_docs = load_drugbank_csv(file_path)
    elif ext == ".pdf":
        raw_docs = load_pdf(file_path)
    else:
        raise ValueError("Unsupported file type: only .csv and .pdf are supported.")

    chunks = chunk_documents(raw_docs)
    return chunks


def chunk_documents(docs, chunk_size=600, chunk_overlap=80):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)
