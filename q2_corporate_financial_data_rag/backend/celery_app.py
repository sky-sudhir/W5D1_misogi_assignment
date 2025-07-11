from celery import Celery
from config import settings
import structlog

logger = structlog.get_logger(__name__)

# Create Celery app
celery_app = Celery(
    "financial_rag_system",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["tasks"]
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Task routing
celery_app.conf.task_routes = {
    "tasks.process_document": {"queue": "document_processing"},
    "tasks.batch_process_documents": {"queue": "batch_processing"},
    "tasks.analyze_financial_data": {"queue": "analysis"},
}

if __name__ == "__main__":
    celery_app.start() 