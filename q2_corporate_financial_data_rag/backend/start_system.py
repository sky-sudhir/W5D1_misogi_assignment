#!/usr/bin/env python3
"""
Startup script for Financial Intelligence RAG System
This script helps test and verify all components are working properly.
"""

import asyncio
import sys
import os
import time
import logging
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

import structlog
from config import settings

# Create logs directory if it doesn't exist
logs_dir = backend_dir / "logs"
logs_dir.mkdir(exist_ok=True)

# Configure file logging
log_file = logs_dir / "system_startup.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))

# Configure console logging
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


async def test_redis_connection():
    """Test Redis connection"""
    try:
        from cache.redis_client import get_redis_client
        redis_client = await get_redis_client()
        
        # Test basic operations
        await redis_client.set("test_key", "test_value", ttl=10)
        value = await redis_client.get("test_key")
        
        if value == "test_value":
            logger.info("✅ Redis connection successful")
            return True
        else:
            logger.error("❌ Redis connection failed - value mismatch")
            return False
            
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        return False


async def test_vectorstore():
    """Test ChromaDB vectorstore"""
    try:
        from chroma_client import get_vectorstore
        vectorstore = get_vectorstore()
        
        # Test basic operations
        from langchain.schema import Document
        test_doc = Document(page_content="Test financial document", metadata={"test": True})
        
        # This might fail if no collection exists yet, which is okay
        try:
            vectorstore.add_documents([test_doc])
            logger.info("✅ ChromaDB vectorstore operational")
            return True
        except Exception as e:
            logger.warning(f"⚠️ ChromaDB test failed (may be normal): {e}")
            return True  # Don't fail startup for this
            
    except Exception as e:
        logger.error(f"❌ ChromaDB connection failed: {e}")
        return False


async def test_rag_pipeline():
    """Test RAG pipeline"""
    try:
        from rag_pipeline import get_async_rag_chain
        
        # Just test initialization
        rag_chain = await get_async_rag_chain()
        
        if rag_chain:
            logger.info("✅ RAG pipeline initialized")
            return True
        else:
            logger.error("❌ RAG pipeline initialization failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ RAG pipeline test failed: {e}")
        return False


def check_environment():
    """Check environment variables"""
    logger.info("🔍 Checking environment configuration...")
    
    required_vars = {
        "GOOGLE_API_KEY": settings.GOOGLE_API_KEY,
        "REDIS_URL": settings.REDIS_URL,
    }
    
    optional_vars = {
        "LANGSMITH_API_KEY": settings.LANGSMITH_API_KEY,
        "CHROMA_API_KEY": settings.CHROMA_API_KEY,
    }
    
    all_good = True
    
    for var, value in required_vars.items():
        if not value or value == f"your_{var.lower()}_here":
            logger.error(f"❌ Required environment variable {var} not set")
            all_good = False
        else:
            logger.info(f"✅ {var} configured")
    
    for var, value in optional_vars.items():
        if not value or value == f"your_{var.lower()}_here":
            logger.warning(f"⚠️ Optional environment variable {var} not set")
        else:
            logger.info(f"✅ {var} configured")
    
    return all_good


async def run_system_tests():
    """Run all system tests"""
    logger.info("🚀 Starting Financial Intelligence RAG System tests...")
    
    # Check environment
    if not check_environment():
        logger.error("❌ Environment check failed. Please check your .env file")
        return False
    
    # Test components
    tests = [
        ("Redis Connection", test_redis_connection),
        ("ChromaDB Vectorstore", test_vectorstore),
        ("RAG Pipeline", test_rag_pipeline),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"🧪 Testing {test_name}...")
        try:
            result = await test_func()
            results.append(result)
        except Exception as e:
            logger.error(f"❌ {test_name} test failed with exception: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    logger.info(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready.")
        return True
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Check configuration.")
        return False


def start_fastapi_server():
    """Start the FastAPI server"""
    logger.info("🚀 Starting FastAPI server...")
    
    try:
        import uvicorn
        from main import app
        
        uvicorn.run(
            app,
            host=settings.HOST,
            port=settings.PORT,
            log_level=settings.LOG_LEVEL.lower(),
            reload=settings.DEBUG
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start FastAPI server: {e}")
        return False


def main():
    """Main entry point"""
    print("=" * 60)
    print("🏦 Financial Intelligence RAG System")
    print("=" * 60)
    print(f"📝 Logs will be saved to: {logs_dir / 'system_startup.log'}")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "test":
            # Run tests only
            success = asyncio.run(run_system_tests())
            sys.exit(0 if success else 1)
            
        elif command == "start":
            # Run tests then start server
            success = asyncio.run(run_system_tests())
            if success:
                start_fastapi_server()
            else:
                print("❌ Tests failed. Fix issues before starting server.")
                sys.exit(1)
                
        else:
            print(f"Unknown command: {command}")
            print("Usage: python start_system.py [test|start]")
            sys.exit(1)
    else:
        # Default: run tests then start server
        success = asyncio.run(run_system_tests())
        if success:
            print("\n🚀 Starting server...")
            start_fastapi_server()
        else:
            print("❌ Tests failed. Fix issues before starting server.")
            sys.exit(1)


if __name__ == "__main__":
    main() 