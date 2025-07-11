import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
import structlog
from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import Response

# Import our modules
from config import settings
from cache.redis_client import get_redis_client, RedisClient
from rag_pipeline import get_async_rag_chain
from chroma_client import get_vectorstore
from ingest import async_ingest_file, chunk_documents
from raga_eval import evaluate_single_sample

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
CACHE_HITS = Counter('cache_hits_total', 'Total cache hits')
CACHE_MISSES = Counter('cache_misses_total', 'Total cache misses')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active connections')
QUERY_PROCESSING_TIME = Histogram('query_processing_seconds', 'Query processing time')

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Global variables
redis_client: Optional[RedisClient] = None
rag_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Financial Intelligence RAG System...")
    
    global redis_client, rag_chain
    
    try:
        # Initialize Redis
        redis_client = await get_redis_client()
        logger.info("Redis client initialized")
        
        # Initialize RAG chain
        rag_chain = await get_async_rag_chain()
        logger.info("RAG chain initialized")
        
        # Initialize LangSmith if configured
        if settings.LANGSMITH_API_KEY:
            import os
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
            os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
            logger.info("LangSmith monitoring enabled")
        
        logger.info("Application startup completed")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Financial Intelligence RAG System...")
    
    if redis_client:
        await redis_client.disconnect()
    
    logger.info("Application shutdown completed")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-scale Financial Intelligence RAG System with Redis caching and concurrent request handling",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# Rate limit exceeded handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Pydantic models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    use_cache: bool = Field(default=True)
    is_realtime: bool = Field(default=True)


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    cached: bool = False
    cache_hit: bool = False
    processing_time: float
    timestamp: str


class FinancialMetricsRequest(BaseModel):
    company: str = Field(..., min_length=1, max_length=100)
    metrics: List[str] = Field(default=["revenue", "profit", "assets", "liabilities"])


class CompanyComparisonRequest(BaseModel):
    companies: List[str] = Field(..., min_items=2, max_items=10)
    metrics: List[str] = Field(default=["revenue", "profit_margin", "market_cap"])


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    redis_connected: bool
    cache_stats: Dict[str, Any]


# Dependency to get Redis client
async def get_redis() -> RedisClient:
    """Get Redis client dependency"""
    global redis_client
    if not redis_client:
        redis_client = await get_redis_client()
    return redis_client


# Middleware for metrics and logging
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware for metrics collection and request logging"""
    start_time = time.time()
    
    # Increment active connections
    ACTIVE_CONNECTIONS.inc()
    
    try:
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        REQUEST_DURATION.observe(duration)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        # Log request
        logger.info(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration=duration,
            client_ip=get_remote_address(request)
        )
        
        return response
        
    except Exception as e:
        # Record error metrics
        duration = time.time() - start_time
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=500
        ).inc()
        
        logger.error(
            "Request failed",
            method=request.method,
            path=request.url.path,
            duration=duration,
            error=str(e),
            client_ip=get_remote_address(request)
        )
        
        raise
    
    finally:
        # Decrement active connections
        ACTIVE_CONNECTIONS.dec()


# API Routes

@app.get("/health", response_model=HealthResponse)
async def health_check(redis: RedisClient = Depends(get_redis)):
    """Health check endpoint"""
    cache_stats = await redis.get_cache_stats()
    
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        redis_connected=redis.redis is not None,
        cache_stats=cache_stats
    )


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")


@app.post("/query", response_model=QueryResponse)
@limiter.limit(f"{settings.RATE_LIMIT_REQUESTS}/{settings.RATE_LIMIT_WINDOW}seconds")
async def query_financial_rag(
    request: Request,
    query_request: QueryRequest,
    redis: RedisClient = Depends(get_redis)
):
    """Main query endpoint with caching and rate limiting"""
    start_time = time.time()
    
    try:
        # Check cache first if enabled
        cached_result = None
        if query_request.use_cache:
            cached_result = await redis.get_cached_query_result(query_request.question)
            
            if cached_result:
                CACHE_HITS.inc()
                processing_time = time.time() - start_time
                
                return QueryResponse(
                    answer=cached_result["result"]["answer"],
                    sources=cached_result["result"]["sources"],
                    cached=True,
                    cache_hit=True,
                    processing_time=processing_time,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC")
                )
        
        # Cache miss - process query
        CACHE_MISSES.inc()
        
        # Process with RAG chain
        with QUERY_PROCESSING_TIME.time():
            result = await rag_chain.ainvoke({"query": query_request.question})
        
        answer = result["result"]
        sources = [doc.page_content for doc in result["source_documents"]]
        
        # Prepare response
        response_data = {
            "answer": answer,
            "sources": sources
        }
        
        # Cache the result if enabled
        if query_request.use_cache:
            await redis.cache_query_result(
                query_request.question,
                response_data,
                query_request.is_realtime
            )
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            cached=False,
            cache_hit=False,
            processing_time=processing_time,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC")
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.post("/financial-metrics")
@limiter.limit(f"{settings.RATE_LIMIT_REQUESTS}/{settings.RATE_LIMIT_WINDOW}seconds")
async def get_financial_metrics(
    request: Request,
    metrics_request: FinancialMetricsRequest,
    redis: RedisClient = Depends(get_redis)
):
    """Get financial metrics for a company"""
    try:
        # Check cache first
        cached_metrics = await redis.get_cached_financial_metrics(metrics_request.company)
        
        if cached_metrics:
            CACHE_HITS.inc()
            return {
                "company": metrics_request.company,
                "metrics": cached_metrics["metrics"],
                "cached": True,
                "timestamp": cached_metrics["cached_at"]
            }
        
        CACHE_MISSES.inc()
        
        # Process financial metrics query
        query = f"What are the financial metrics for {metrics_request.company}? Include: {', '.join(metrics_request.metrics)}"
        
        result = await rag_chain.ainvoke({"query": query})
        
        # Parse and structure the metrics (simplified for demo)
        metrics_data = {
            "raw_answer": result["result"],
            "requested_metrics": metrics_request.metrics,
            "sources": [doc.page_content[:200] for doc in result["source_documents"]]
        }
        
        # Cache the result
        await redis.cache_financial_metrics(metrics_request.company, metrics_data)
        
        return {
            "company": metrics_request.company,
            "metrics": metrics_data,
            "cached": False,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC")
        }
        
    except Exception as e:
        logger.error(f"Financial metrics processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Financial metrics processing failed: {str(e)}")


@app.post("/company-comparison")
@limiter.limit(f"{settings.RATE_LIMIT_REQUESTS}/{settings.RATE_LIMIT_WINDOW}seconds")
async def compare_companies(
    request: Request,
    comparison_request: CompanyComparisonRequest,
    redis: RedisClient = Depends(get_redis)
):
    """Compare financial metrics between companies"""
    try:
        companies_str = ", ".join(comparison_request.companies)
        metrics_str = ", ".join(comparison_request.metrics)
        
        query = f"Compare the financial performance of {companies_str} focusing on {metrics_str}. Provide a detailed comparison."
        
        # Check cache
        cached_result = await redis.get_cached_query_result(query)
        
        if cached_result:
            CACHE_HITS.inc()
            return {
                "companies": comparison_request.companies,
                "metrics": comparison_request.metrics,
                "comparison": cached_result["result"]["answer"],
                "sources": cached_result["result"]["sources"],
                "cached": True,
                "timestamp": cached_result["cached_at"]
            }
        
        CACHE_MISSES.inc()
        
        # Process comparison
        result = await rag_chain.ainvoke({"query": query})
        
        response_data = {
            "answer": result["result"],
            "sources": [doc.page_content for doc in result["source_documents"]]
        }
        
        # Cache the result
        await redis.cache_query_result(query, response_data, is_realtime=False)
        
        return {
            "companies": comparison_request.companies,
            "metrics": comparison_request.metrics,
            "comparison": result["result"],
            "sources": [doc.page_content[:200] for doc in result["source_documents"]],
            "cached": False,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC")
        }
        
    except Exception as e:
        logger.error(f"Company comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Company comparison failed: {str(e)}")


@app.post("/ingest")
@limiter.limit("10/hour")
async def ingest_financial_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    redis: RedisClient = Depends(get_redis)
):
    """Ingest financial document with background processing"""
    try:
        # Validate file
        if not file.filename.lower().endswith(('.pdf', '.csv')):
            raise HTTPException(status_code=400, detail="Only PDF and CSV files are supported")
        
        if file.size and file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File size exceeds maximum limit")
        
        # Save file temporarily
        contents = await file.read()
        temp_path = f"./backend/data/{file.filename}"
        
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # Add background task for processing
        background_tasks.add_task(process_document_background, temp_path, file.filename)
        
        return {
            "message": f"Document {file.filename} queued for processing",
            "filename": file.filename,
            "size": len(contents),
            "status": "queued"
        }
        
    except Exception as e:
        logger.error(f"Document ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {str(e)}")


async def process_document_background(file_path: str, filename: str):
    """Background task for document processing"""
    try:
        logger.info(f"Processing document: {filename}")
        
        # Process document
        docs = await async_ingest_file(file_path)
        chunks = chunk_documents(docs)
        
        # Add to vector store
        vectorstore = get_vectorstore()
        await vectorstore.aadd_documents(chunks)
        
        # Clean up
        import os
        os.remove(file_path)
        
        logger.info(f"Document processed successfully: {filename}, {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Background document processing failed: {e}")


@app.get("/cache/stats")
async def get_cache_statistics(redis: RedisClient = Depends(get_redis)):
    """Get cache statistics"""
    try:
        stats = await redis.get_cache_stats()
        return {
            "cache_stats": stats,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC")
        }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cache statistics")


@app.delete("/cache/clear")
async def clear_cache(
    pattern: Optional[str] = None,
    redis: RedisClient = Depends(get_redis)
):
    """Clear cache (admin endpoint)"""
    try:
        if pattern:
            cleared = await redis.clear_cache_pattern(f"{settings.CACHE_KEY_PREFIX}:{pattern}*")
        else:
            cleared = await redis.clear_cache_pattern(f"{settings.CACHE_KEY_PREFIX}:*")
        
        return {
            "message": f"Cleared {cleared} cache entries",
            "pattern": pattern or "all",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC")
        }
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    ) 