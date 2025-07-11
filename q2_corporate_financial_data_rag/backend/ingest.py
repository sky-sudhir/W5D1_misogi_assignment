
import asyncio
import pandas as pd
import fitz  # PyMuPDF
import pdfplumber
import os
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import settings
import structlog

logger = structlog.get_logger(__name__)


# Enhanced PDF processing for financial documents
async def load_financial_pdf(file_path: str) -> List[Document]:
    """Load and parse financial PDF documents with enhanced extraction"""
    try:
        documents = []
        
        # Use pdfplumber for better text extraction
        with pdfplumber.open(file_path) as pdf:
            full_text = ""
            metadata = {
                "source": file_path,
                "total_pages": len(pdf.pages),
                "document_type": "financial_report"
            }
            
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    
                    # Extract tables if present
                    tables = page.extract_tables()
                    if tables:
                        for table_idx, table in enumerate(tables):
                            table_text = "\n".join(["\t".join(row) for row in table if row])
                            full_text += f"\n--- Table {table_idx + 1} on Page {page_num + 1} ---\n{table_text}"
            
            # Create document with metadata
            doc = Document(
                page_content=full_text,
                metadata=metadata
            )
            documents.append(doc)
            
        logger.info(f"Successfully processed PDF: {file_path}, {len(documents)} documents extracted")
        return documents
        
    except Exception as e:
        logger.error(f"Failed to process PDF {file_path}: {e}")
        # Fallback to PyMuPDF
        return await load_pdf_fallback(file_path)


async def load_pdf_fallback(file_path: str) -> List[Document]:
    """Fallback PDF processing using PyMuPDF"""
    try:
        doc = fitz.open(file_path)
        full_text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        
        doc.close()
        
        metadata = {
            "source": file_path,
            "document_type": "financial_report",
            "processing_method": "fallback"
        }
        
        return [Document(page_content=full_text, metadata=metadata)]
        
    except Exception as e:
        logger.error(f"Fallback PDF processing failed for {file_path}: {e}")
        raise


# Enhanced CSV processing for financial data
async def load_financial_csv(file_path: str) -> List[Document]:
    """Load and process financial CSV data"""
    try:
        df = pd.read_csv(file_path)
        documents = []
        
        # Process each row as a separate document
        for idx, row in df.iterrows():
            # Create structured content
            content_parts = []
            for col in df.columns:
                if pd.notnull(row[col]):
                    content_parts.append(f"{col}: {row[col]}")
            
            content = "\n".join(content_parts)
            
            metadata = {
                "source": file_path,
                "row_index": idx,
                "document_type": "financial_data",
                "columns": list(df.columns)
            }
            
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        
        logger.info(f"Successfully processed CSV: {file_path}, {len(documents)} documents extracted")
        return documents
        
    except Exception as e:
        logger.error(f"Failed to process CSV {file_path}: {e}")
        raise


# Async file processing dispatcher
async def async_ingest_file(file_path: str) -> List[Document]:
    """Async file ingestion dispatcher"""
    try:
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".pdf":
            return await load_financial_pdf(file_path)
        elif ext == ".csv":
            return await load_financial_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Only .pdf and .csv are supported.")
            
    except Exception as e:
        logger.error(f"Async file ingestion failed for {file_path}: {e}")
        raise


# Synchronous wrapper for backward compatibility
def ingest_file(file_path: str) -> List[Document]:
    """Synchronous wrapper for file ingestion"""
    try:
        return asyncio.run(async_ingest_file(file_path))
    except Exception as e:
        logger.error(f"Synchronous file ingestion failed for {file_path}: {e}")
        raise


# Enhanced chunking for financial documents
def chunk_documents(
    docs: List[Document], 
    chunk_size: int = None, 
    chunk_overlap: int = None
) -> List[Document]:
    """Enhanced document chunking for financial documents"""
    try:
        chunk_size = chunk_size or settings.CHUNK_SIZE
        chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        # Create splitter with financial document-specific separators
        financial_separators = [
            "\n--- Table",  # Table separators
            "\n--- Page",   # Page separators
            "\n\n",         # Double newlines
            "\n",           # Single newlines
            ". ",           # Sentences
            ", ",           # Clauses
            " ",            # Words
            ""              # Characters
        ]
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=financial_separators,
            length_function=len,
            is_separator_regex=False
        )
        
        chunks = splitter.split_documents(docs)
        
        # Enhance chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "chunk_size": len(chunk.page_content),
                "processing_timestamp": pd.Timestamp.now().isoformat()
            })
        
        logger.info(f"Successfully chunked documents: {len(chunks)} chunks created")
        return chunks
        
    except Exception as e:
        logger.error(f"Document chunking failed: {e}")
        raise


# Financial document analysis
async def analyze_document_content(file_path: str) -> Dict[str, Any]:
    """Analyze financial document content and extract metadata"""
    try:
        docs = await async_ingest_file(file_path)
        
        analysis = {
            "file_path": file_path,
            "document_count": len(docs),
            "total_length": sum(len(doc.page_content) for doc in docs),
            "document_types": list(set(doc.metadata.get("document_type", "unknown") for doc in docs)),
            "has_tables": any("Table" in doc.page_content for doc in docs),
            "estimated_chunks": 0
        }
        
        # Estimate chunks
        chunks = chunk_documents(docs)
        analysis["estimated_chunks"] = len(chunks)
        
        # Extract potential financial keywords
        financial_keywords = [
            "revenue", "profit", "loss", "assets", "liabilities", "equity",
            "cash flow", "earnings", "dividend", "market cap", "debt",
            "financial statement", "balance sheet", "income statement"
        ]
        
        content_lower = " ".join(doc.page_content.lower() for doc in docs)
        found_keywords = [kw for kw in financial_keywords if kw in content_lower]
        analysis["financial_keywords"] = found_keywords
        analysis["financial_relevance_score"] = len(found_keywords) / len(financial_keywords)
        
        logger.info(f"Document analysis completed for {file_path}")
        return analysis
        
    except Exception as e:
        logger.error(f"Document analysis failed for {file_path}: {e}")
        raise


# Batch processing for multiple documents
async def batch_ingest_documents(file_paths: List[str]) -> Dict[str, Any]:
    """Process multiple documents in batch"""
    try:
        results = {
            "processed": [],
            "failed": [],
            "total_chunks": 0,
            "processing_time": 0
        }
        
        import time
        start_time = time.time()
        
        # Process documents concurrently
        tasks = [async_ingest_file(path) for path in file_paths]
        document_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(document_results):
            file_path = file_paths[i]
            
            if isinstance(result, Exception):
                results["failed"].append({
                    "file_path": file_path,
                    "error": str(result)
                })
                logger.error(f"Failed to process {file_path}: {result}")
            else:
                chunks = chunk_documents(result)
                results["processed"].append({
                    "file_path": file_path,
                    "document_count": len(result),
                    "chunk_count": len(chunks)
                })
                results["total_chunks"] += len(chunks)
                logger.info(f"Successfully processed {file_path}: {len(chunks)} chunks")
        
        results["processing_time"] = time.time() - start_time
        
        logger.info(f"Batch processing completed: {len(results['processed'])} successful, {len(results['failed'])} failed")
        return results
        
    except Exception as e:
        logger.error(f"Batch document processing failed: {e}")
        raise


# Legacy functions for backward compatibility
def load_drugbank_csv(file_path: str) -> List[Document]:
    """Legacy CSV loading function"""
    return asyncio.run(load_financial_csv(file_path))


def load_pdf(file_path: str) -> List[Document]:
    """Legacy PDF loading function"""
    return asyncio.run(load_financial_pdf(file_path))
