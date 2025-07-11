import os
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Production settings for Financial Intelligence RAG System"""
    
    # Application Settings
    APP_NAME: str = "Financial Intelligence RAG System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Server Settings
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=4, env="WORKERS")
    
    # Database Settings
    DATABASE_URL: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # Redis Settings
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    # Cache Settings
    CACHE_TTL_REALTIME: int = Field(default=3600, env="CACHE_TTL_REALTIME")  # 1 hour
    CACHE_TTL_HISTORICAL: int = Field(default=86400, env="CACHE_TTL_HISTORICAL")  # 24 hours
    CACHE_KEY_PREFIX: str = Field(default="finrag", env="CACHE_KEY_PREFIX")
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_WINDOW: int = Field(default=3600, env="RATE_LIMIT_WINDOW")  # 1 hour
    RATE_LIMIT_ENABLED: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    
    # Concurrency Settings
    MAX_CONCURRENT_REQUESTS: int = Field(default=200, env="MAX_CONCURRENT_REQUESTS")
    REQUEST_TIMEOUT: int = Field(default=30, env="REQUEST_TIMEOUT")
    
    # Vector Store Settings
    CHROMA_API_KEY: Optional[str] = Field(default=None, env="CHROMA_API_KEY")
    CHROMA_TENANT: Optional[str] = Field(default=None, env="CHROMA_TENANT")
    CHROMA_DB: Optional[str] = Field(default=None, env="CHROMA_DB")
    COLLECTION_NAME: str = Field(default="financial_documents", env="COLLECTION_NAME")
    
    # LLM Settings
    GOOGLE_API_KEY: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    MODEL_NAME: str = Field(default="gemini-2.0-flash", env="MODEL_NAME")
    MODEL_TEMPERATURE: float = Field(default=0.0, env="MODEL_TEMPERATURE")
    MAX_TOKENS: Optional[int] = Field(default=None, env="MAX_TOKENS")
    
    # Monitoring Settings
    LANGSMITH_API_KEY: Optional[str] = Field(default=None, env="LANGSMITH_API_KEY")
    LANGSMITH_PROJECT: str = Field(default="financial-rag-system", env="LANGSMITH_PROJECT")
    LANGSMITH_ENDPOINT: str = Field(default="https://api.smith.langchain.com", env="LANGSMITH_ENDPOINT")
    
    # Metrics and Monitoring
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=8001, env="METRICS_PORT")
    
    # Background Jobs
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/1", env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/2", env="CELERY_RESULT_BACKEND")
    
    # Document Processing
    CHUNK_SIZE: int = Field(default=1000, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(default=200, env="CHUNK_OVERLAP")
    MAX_FILE_SIZE: int = Field(default=50 * 1024 * 1024, env="MAX_FILE_SIZE")  # 50MB
    
    # Financial Data Settings
    SUPPORTED_FILE_TYPES: List[str] = Field(default=["pdf", "csv"], env="SUPPORTED_FILE_TYPES")
    FINANCIAL_METRICS_CACHE_TTL: int = Field(default=7200, env="FINANCIAL_METRICS_CACHE_TTL")  # 2 hours
    
    # Security Settings
    SECRET_KEY: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    ALGORITHM: str = Field(default="HS256", env="ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # CORS Settings
    CORS_ORIGINS: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    CORS_ALLOW_METHODS: List[str] = Field(default=["*"], env="CORS_ALLOW_METHODS")
    CORS_ALLOW_HEADERS: List[str] = Field(default=["*"], env="CORS_ALLOW_HEADERS")
    
    # Logging Settings
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings() 