# Financial Intelligence RAG System

A production-scale Financial Intelligence RAG (Retrieval-Augmented Generation) System designed to handle concurrent queries on corporate financial reports and earnings data with Redis caching and OpenAI API integration.

## 🚀 Features

### Core Capabilities
- **Financial Document Processing**: Enhanced PDF and CSV parsing for financial reports
- **Intelligent Query Processing**: Financial-specific RAG pipeline with Google Gemini
- **Real-time Caching**: Redis-based caching with TTL management (1h real-time, 24h historical)
- **Concurrent Request Handling**: Supports 200+ concurrent users with <2s response time
- **Background Processing**: Celery-based async document ingestion and analysis
- **Performance Monitoring**: LangSmith integration and Prometheus metrics

### Production Features
- **Rate Limiting**: Per-API-key rate limiting and request throttling
- **Load Balancing**: Async FastAPI with connection pooling
- **Monitoring**: Real-time system metrics and performance dashboards
- **Scalability**: Horizontal scaling support with Redis cluster
- **Security**: JWT authentication, CORS, and input validation

## 📋 Requirements

- Python 3.8+
- Redis Server
- Google Gemini API Key
- LangSmith API Key (optional)
- ChromaDB (local or cloud)

## 🛠️ Installation

### 1. Clone and Setup
```bash
git clone <repository-url>
cd financial-rag-system
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Configuration
```bash
# Copy the example environment file
cp backend/env_example.txt .env

# Edit .env with your configuration
# Required settings:
# - GOOGLE_API_KEY: Your Google Gemini API key
# - REDIS_URL: Redis connection string
# - LANGSMITH_API_KEY: LangSmith API key (optional)
```

### 3. Start Redis Server
```bash
# Option 1: Docker
docker run -d -p 6379:6379 redis:alpine

# Option 2: Local installation
redis-server
```

### 4. Start the Application
```bash
# Development mode
python backend/main.py

# Production mode
gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### 5. Start Background Workers (Optional)
```bash
# Start Celery worker
celery -A backend.celery_app worker --loglevel=info

# Start Celery beat (for scheduled tasks)
celery -A backend.celery_app beat --loglevel=info
```

## 📊 API Endpoints

### Core Endpoints

#### Query Financial Data
```http
POST /query
Content-Type: application/json

{
  "question": "What is Apple's revenue for 2023?",
  "use_cache": true,
  "is_realtime": true
}
```

#### Financial Metrics
```http
POST /financial-metrics
Content-Type: application/json

{
  "company": "Apple",
  "metrics": ["revenue", "profit", "assets", "liabilities"]
}
```

#### Company Comparison
```http
POST /company-comparison
Content-Type: application/json

{
  "companies": ["Apple", "Microsoft", "Google"],
  "metrics": ["revenue", "profit_margin", "market_cap"]
}
```

#### Document Ingestion
```http
POST /ingest
Content-Type: multipart/form-data

file: <financial_report.pdf>
```

### Monitoring Endpoints

#### Health Check
```http
GET /health
```

#### Cache Statistics
```http
GET /cache/stats
```

#### Prometheus Metrics
```http
GET /metrics
```

## 🗂️ Project Structure

```
financial-rag-system/
├── backend/
│   ├── cache/
│   │   └── redis_client.py      # Redis caching layer
│   ├── config.py                # Configuration management
│   ├── main.py                  # FastAPI application
│   ├── rag_pipeline.py          # RAG processing pipeline
│   ├── ingest.py                # Document ingestion
│   ├── chroma_client.py         # Vector database client
│   ├── celery_app.py            # Celery configuration
│   ├── tasks.py                 # Background tasks
│   ├── load_testing.py          # Load testing script
│   └── env_example.txt          # Environment configuration
├── ui/
│   └── app.py                   # Streamlit UI
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🧪 Testing

### Load Testing
```bash
# Run load test with 200 concurrent users for 10 minutes
python backend/load_testing.py

# Or use Locust directly
locust -f backend/load_testing.py --host http://localhost:8000 --users 200 --spawn-rate 10 --run-time 10m
```

### Unit Tests
```bash
# Run tests
pytest

# Run with coverage
pytest --cov=backend
```

## 📈 Performance Targets

- **Response Time**: < 2 seconds average
- **Concurrent Users**: 200+ simultaneous users
- **Cache Hit Ratio**: > 70%
- **Uptime**: 99.9%
- **Error Rate**: < 5%

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google Gemini API key | Required |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `LANGSMITH_API_KEY` | LangSmith monitoring key | Optional |
| `RATE_LIMIT_REQUESTS` | Requests per hour | 100 |
| `MAX_CONCURRENT_REQUESTS` | Max concurrent requests | 200 |
| `CACHE_TTL_REALTIME` | Real-time cache TTL (seconds) | 3600 |
| `CACHE_TTL_HISTORICAL` | Historical cache TTL (seconds) | 86400 |

### Redis Configuration

```bash
# Redis databases used:
# 0: Main cache
# 1: Celery broker
# 2: Celery results
```

## 🚀 Deployment

### Docker Deployment
```bash
# Build image
docker build -t financial-rag-system .

# Run with docker-compose
docker-compose up -d
```

### Production Deployment
```bash
# Install production dependencies
pip install gunicorn

# Start with multiple workers
gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

# Start with Celery workers
celery -A backend.celery_app worker --loglevel=info --concurrency=4
```

## 📊 Monitoring

### Prometheus Metrics
- Request count and duration
- Cache hit/miss ratios
- Active connections
- Query processing time

### LangSmith Integration
- Request tracing
- Performance monitoring
- Error tracking
- Response quality metrics

### Health Checks
- Application health: `GET /health`
- Cache statistics: `GET /cache/stats`
- Metrics endpoint: `GET /metrics`

## 🔍 Usage Examples

### Basic Financial Query
```python
import requests

response = requests.post("http://localhost:8000/query", json={
    "question": "What was Apple's revenue in Q3 2023?",
    "use_cache": True
})

print(response.json())
```

### Company Comparison
```python
response = requests.post("http://localhost:8000/company-comparison", json={
    "companies": ["Apple", "Microsoft"],
    "metrics": ["revenue", "profit_margin"]
})

print(response.json())
```

### Document Upload
```python
with open("financial_report.pdf", "rb") as f:
    response = requests.post("http://localhost:8000/ingest", files={"file": f})

print(response.json())
```

## 🐛 Troubleshooting

### Common Issues

1. **Redis Connection Error**
   ```bash
   # Check Redis status
   redis-cli ping
   
   # Restart Redis
   redis-server
   ```

2. **High Memory Usage**
   ```bash
   # Clear cache
   curl -X DELETE http://localhost:8000/cache/clear
   ```

3. **Slow Response Times**
   ```bash
   # Check cache hit ratio
   curl http://localhost:8000/cache/stats
   ```

### Performance Optimization

1. **Increase Cache TTL** for historical data
2. **Scale Redis** with clustering
3. **Add more workers** for concurrent processing
4. **Optimize chunk sizes** for better retrieval

## 📝 API Documentation

Once the application is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🔗 Links

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Redis Documentation](https://redis.io/documentation)
- [LangChain Documentation](https://python.langchain.com/)
- [Google Gemini API](https://ai.google.dev/)
- [LangSmith](https://smith.langchain.com/)

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the API documentation
3. Check application logs
4. Open an issue on GitHub 