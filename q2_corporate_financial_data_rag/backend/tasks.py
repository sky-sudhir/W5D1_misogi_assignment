import os
import asyncio
from typing import List, Dict, Any
from celery import current_task
from celery_app import celery_app
from ingest import async_ingest_file, chunk_documents, analyze_document_content
from chroma_client import get_vectorstore
from rag_pipeline import extract_financial_metrics, analyze_financial_trends
from cache.redis_client import get_redis_client
import structlog

logger = structlog.get_logger(__name__)


@celery_app.task(bind=True)
def process_document(self, file_path: str, filename: str) -> Dict[str, Any]:
    """Background task to process a single document"""
    try:
        # Update task state
        self.update_state(
            state="PROCESSING",
            meta={"filename": filename, "status": "Processing document"}
        )
        
        # Process document asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Ingest document
            docs = loop.run_until_complete(async_ingest_file(file_path))
            
            # Update state
            self.update_state(
                state="PROCESSING",
                meta={"filename": filename, "status": "Chunking document", "docs_count": len(docs)}
            )
            
            # Chunk documents
            chunks = chunk_documents(docs)
            
            # Update state
            self.update_state(
                state="PROCESSING",
                meta={"filename": filename, "status": "Adding to vector store", "chunks_count": len(chunks)}
            )
            
            # Add to vector store
            vectorstore = get_vectorstore()
            vectorstore.add_documents(chunks)
            
            # Clean up file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            result = {
                "filename": filename,
                "status": "completed",
                "docs_processed": len(docs),
                "chunks_created": len(chunks),
                "message": f"Successfully processed {filename}"
            }
            
            logger.info(f"Document processing completed: {filename}")
            return result
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Document processing failed for {filename}: {e}")
        
        # Clean up file on error
        if os.path.exists(file_path):
            os.remove(file_path)
        
        self.update_state(
            state="FAILURE",
            meta={"filename": filename, "error": str(e)}
        )
        
        raise


@celery_app.task(bind=True)
def batch_process_documents(self, file_paths: List[str]) -> Dict[str, Any]:
    """Background task to process multiple documents"""
    try:
        # Update task state
        self.update_state(
            state="PROCESSING",
            meta={"status": "Starting batch processing", "total_files": len(file_paths)}
        )
        
        # Process documents asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Process all documents
            from ingest import batch_ingest_documents
            results = loop.run_until_complete(batch_ingest_documents(file_paths))
            
            # Update state
            self.update_state(
                state="PROCESSING",
                meta={"status": "Adding to vector store", "processed": len(results["processed"])}
            )
            
            # Add all chunks to vector store
            vectorstore = get_vectorstore()
            total_chunks = 0
            
            for file_path in file_paths:
                try:
                    docs = loop.run_until_complete(async_ingest_file(file_path))
                    chunks = chunk_documents(docs)
                    vectorstore.add_documents(chunks)
                    total_chunks += len(chunks)
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
            
            # Clean up files
            for file_path in file_paths:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            result = {
                "status": "completed",
                "total_files": len(file_paths),
                "processed_files": len(results["processed"]),
                "failed_files": len(results["failed"]),
                "total_chunks": total_chunks,
                "processing_time": results["processing_time"]
            }
            
            logger.info(f"Batch processing completed: {result}")
            return result
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        
        # Clean up files on error
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        self.update_state(
            state="FAILURE",
            meta={"error": str(e)}
        )
        
        raise


@celery_app.task(bind=True)
def analyze_financial_data(self, file_path: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
    """Background task to analyze financial document content"""
    try:
        # Update task state
        self.update_state(
            state="PROCESSING",
            meta={"status": "Analyzing document content", "analysis_type": analysis_type}
        )
        
        # Analyze document asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Analyze document content
            analysis = loop.run_until_complete(analyze_document_content(file_path))
            
            # Update state
            self.update_state(
                state="PROCESSING",
                meta={"status": "Generating insights", "financial_score": analysis.get("financial_relevance_score", 0)}
            )
            
            # Generate additional insights based on analysis type
            if analysis_type == "comprehensive":
                # Extract key financial metrics if available
                if analysis.get("financial_relevance_score", 0) > 0.5:
                    # This is a financial document, extract more details
                    insights = {
                        "document_type": "financial_report",
                        "key_metrics_detected": analysis.get("financial_keywords", []),
                        "has_financial_tables": analysis.get("has_tables", False),
                        "content_quality": "high" if analysis.get("financial_relevance_score", 0) > 0.7 else "medium",
                        "recommended_queries": generate_recommended_queries(analysis)
                    }
                else:
                    insights = {
                        "document_type": "non_financial",
                        "content_quality": "low",
                        "recommendation": "This document may not contain relevant financial information"
                    }
                
                analysis["insights"] = insights
            
            result = {
                "status": "completed",
                "analysis": analysis,
                "analysis_type": analysis_type
            }
            
            logger.info(f"Financial analysis completed for {file_path}")
            return result
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Financial analysis failed for {file_path}: {e}")
        
        self.update_state(
            state="FAILURE",
            meta={"error": str(e)}
        )
        
        raise


@celery_app.task(bind=True)
def extract_company_metrics(self, company: str, metrics: List[str]) -> Dict[str, Any]:
    """Background task to extract financial metrics for a company"""
    try:
        # Update task state
        self.update_state(
            state="PROCESSING",
            meta={"status": "Extracting financial metrics", "company": company}
        )
        
        # Extract metrics asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Extract financial metrics
            result = loop.run_until_complete(extract_financial_metrics(company, metrics))
            
            # Cache the result
            redis_client = loop.run_until_complete(get_redis_client())
            loop.run_until_complete(redis_client.cache_financial_metrics(company, result))
            
            logger.info(f"Financial metrics extracted for {company}")
            return result
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Financial metrics extraction failed for {company}: {e}")
        
        self.update_state(
            state="FAILURE",
            meta={"error": str(e)}
        )
        
        raise


@celery_app.task(bind=True)
def analyze_company_trends(self, company: str, time_period: str = "quarterly") -> Dict[str, Any]:
    """Background task to analyze financial trends for a company"""
    try:
        # Update task state
        self.update_state(
            state="PROCESSING",
            meta={"status": "Analyzing financial trends", "company": company}
        )
        
        # Analyze trends asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Analyze financial trends
            result = loop.run_until_complete(analyze_financial_trends(company, time_period))
            
            # Cache the result
            redis_client = loop.run_until_complete(get_redis_client())
            cache_key = f"trends_{company}_{time_period}"
            loop.run_until_complete(redis_client.set(cache_key, result, ttl=7200))  # 2 hours
            
            logger.info(f"Financial trends analyzed for {company}")
            return result
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Financial trends analysis failed for {company}: {e}")
        
        self.update_state(
            state="FAILURE",
            meta={"error": str(e)}
        )
        
        raise


def generate_recommended_queries(analysis: Dict[str, Any]) -> List[str]:
    """Generate recommended queries based on document analysis"""
    queries = []
    
    keywords = analysis.get("financial_keywords", [])
    
    if "revenue" in keywords:
        queries.append("What is the company's revenue trend?")
    
    if "profit" in keywords:
        queries.append("What is the company's profitability?")
    
    if "cash flow" in keywords:
        queries.append("How is the company's cash flow performance?")
    
    if "debt" in keywords:
        queries.append("What is the company's debt situation?")
    
    if "assets" in keywords:
        queries.append("What are the company's key assets?")
    
    # Add generic queries if no specific keywords found
    if not queries:
        queries = [
            "What are the key financial highlights?",
            "What are the main financial risks?",
            "What is the overall financial health?"
        ]
    
    return queries[:5]  # Return top 5 queries


# Task monitoring
@celery_app.task
def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get status of a background task"""
    try:
        result = celery_app.AsyncResult(task_id)
        
        return {
            "task_id": task_id,
            "state": result.state,
            "result": result.result,
            "info": result.info
        }
        
    except Exception as e:
        logger.error(f"Failed to get task status for {task_id}: {e}")
        return {
            "task_id": task_id,
            "state": "ERROR",
            "error": str(e)
        } 