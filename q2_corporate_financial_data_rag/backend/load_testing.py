import time
import random
import json
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner
import structlog

logger = structlog.get_logger(__name__)


class FinancialRAGUser(HttpUser):
    """Load testing user for Financial Intelligence RAG System"""
    
    # Wait between 1-5 seconds between requests
    wait_time = between(1, 5)
    
    def on_start(self):
        """Initialize user session"""
        self.client.verify = False  # Disable SSL verification for testing
        self.queries = [
            "What is Apple's revenue for 2023?",
            "Compare Microsoft and Google's profitability",
            "What are Tesla's financial metrics?",
            "Analyze Amazon's cash flow trends",
            "What is the debt-to-equity ratio for Apple?",
            "Compare the market cap of tech companies",
            "What are the key financial risks for Microsoft?",
            "Analyze Google's revenue growth over time",
            "What is Netflix's subscriber growth trend?",
            "Compare the financial performance of FAANG companies",
            "What are the quarterly earnings for Apple?",
            "Analyze the financial health of Tesla",
            "What is Amazon's operating margin?",
            "Compare the ROI of tech companies",
            "What are the investment opportunities in tech stocks?"
        ]
        
        self.companies = [
            "Apple", "Microsoft", "Google", "Amazon", "Tesla", 
            "Netflix", "Meta", "Nvidia", "Adobe", "Salesforce"
        ]
        
        self.metrics = [
            "revenue", "profit", "assets", "liabilities", "equity",
            "cash_flow", "debt", "market_cap", "ROI", "profit_margin"
        ]
    
    @task(5)
    def query_financial_rag(self):
        """Main query endpoint test"""
        query = random.choice(self.queries)
        
        payload = {
            "question": query,
            "use_cache": random.choice([True, False]),
            "is_realtime": random.choice([True, False])
        }
        
        with self.client.post("/query", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_fields = ["answer", "sources", "processing_time", "timestamp"]
                if all(field in data for field in required_fields):
                    # Check response time
                    if response.elapsed.total_seconds() < 2.0:
                        response.success()
                        
                        # Log cache performance
                        if data.get("cache_hit"):
                            logger.info(f"Cache hit for query: {query[:30]}...")
                        else:
                            logger.info(f"Cache miss for query: {query[:30]}...")
                    else:
                        response.failure(f"Response time too slow: {response.elapsed.total_seconds():.2f}s")
                else:
                    response.failure(f"Missing required fields in response: {data.keys()}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
    
    @task(2)
    def get_financial_metrics(self):
        """Financial metrics endpoint test"""
        company = random.choice(self.companies)
        selected_metrics = random.sample(self.metrics, k=random.randint(2, 5))
        
        payload = {
            "company": company,
            "metrics": selected_metrics
        }
        
        with self.client.post("/financial-metrics", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                
                if "company" in data and "metrics" in data:
                    response.success()
                    logger.info(f"Financial metrics retrieved for {company}")
                else:
                    response.failure(f"Invalid response structure: {data.keys()}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
    
    @task(1)
    def compare_companies(self):
        """Company comparison endpoint test"""
        selected_companies = random.sample(self.companies, k=random.randint(2, 4))
        selected_metrics = random.sample(self.metrics, k=random.randint(2, 4))
        
        payload = {
            "companies": selected_companies,
            "metrics": selected_metrics
        }
        
        with self.client.post("/company-comparison", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                
                if "companies" in data and "comparison" in data:
                    response.success()
                    logger.info(f"Company comparison completed for {selected_companies}")
                else:
                    response.failure(f"Invalid response structure: {data.keys()}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
    
    @task(1)
    def health_check(self):
        """Health check endpoint test"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure(f"Unhealthy status: {data.get('status')}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
    
    @task(1)
    def cache_stats(self):
        """Cache statistics endpoint test"""
        with self.client.get("/cache/stats", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                
                if "cache_stats" in data:
                    response.success()
                    
                    # Log cache hit ratio
                    cache_stats = data["cache_stats"]
                    hit_ratio = cache_stats.get("hit_ratio", 0)
                    logger.info(f"Cache hit ratio: {hit_ratio}%")
                else:
                    response.failure(f"Invalid response structure: {data.keys()}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")


# Performance monitoring
class PerformanceMonitor:
    """Monitor performance metrics during load testing"""
    
    def __init__(self):
        self.response_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.error_count = 0
        self.total_requests = 0
        
    def record_response(self, response_time, cached=False):
        """Record response metrics"""
        self.response_times.append(response_time)
        self.total_requests += 1
        
        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def record_error(self):
        """Record error"""
        self.error_count += 1
        self.total_requests += 1
    
    def get_stats(self):
        """Get performance statistics"""
        if not self.response_times:
            return {"error": "No response times recorded"}
        
        avg_response_time = sum(self.response_times) / len(self.response_times)
        max_response_time = max(self.response_times)
        min_response_time = min(self.response_times)
        
        # Calculate percentiles
        sorted_times = sorted(self.response_times)
        p95_index = int(0.95 * len(sorted_times))
        p99_index = int(0.99 * len(sorted_times))
        
        cache_hit_ratio = (self.cache_hits / (self.cache_hits + self.cache_misses)) * 100 if (self.cache_hits + self.cache_misses) > 0 else 0
        error_rate = (self.error_count / self.total_requests) * 100 if self.total_requests > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "avg_response_time": round(avg_response_time, 3),
            "max_response_time": round(max_response_time, 3),
            "min_response_time": round(min_response_time, 3),
            "p95_response_time": round(sorted_times[p95_index], 3),
            "p99_response_time": round(sorted_times[p99_index], 3),
            "cache_hit_ratio": round(cache_hit_ratio, 2),
            "error_rate": round(error_rate, 2),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "errors": self.error_count
        }


# Global performance monitor
performance_monitor = PerformanceMonitor()


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, context, exception, **kwargs):
    """Event listener for request monitoring"""
    if exception:
        performance_monitor.record_error()
    else:
        # Check if response indicates cache hit
        cached = False
        if hasattr(response, 'json'):
            try:
                data = response.json()
                cached = data.get("cached", False) or data.get("cache_hit", False)
            except:
                pass
        
        performance_monitor.record_response(response_time / 1000, cached)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Event listener for test completion"""
    stats = performance_monitor.get_stats()
    
    print("\n" + "="*50)
    print("LOAD TEST RESULTS")
    print("="*50)
    print(f"Total Requests: {stats.get('total_requests', 0)}")
    print(f"Average Response Time: {stats.get('avg_response_time', 0)} seconds")
    print(f"95th Percentile: {stats.get('p95_response_time', 0)} seconds")
    print(f"99th Percentile: {stats.get('p99_response_time', 0)} seconds")
    print(f"Cache Hit Ratio: {stats.get('cache_hit_ratio', 0)}%")
    print(f"Error Rate: {stats.get('error_rate', 0)}%")
    print("="*50)
    
    # Check performance targets
    targets_met = []
    targets_failed = []
    
    if stats.get('avg_response_time', 0) < 2.0:
        targets_met.append("✓ Average response time < 2 seconds")
    else:
        targets_failed.append("✗ Average response time >= 2 seconds")
    
    if stats.get('cache_hit_ratio', 0) >= 70:
        targets_met.append("✓ Cache hit ratio >= 70%")
    else:
        targets_failed.append("✗ Cache hit ratio < 70%")
    
    if stats.get('error_rate', 0) < 5:
        targets_met.append("✓ Error rate < 5%")
    else:
        targets_failed.append("✗ Error rate >= 5%")
    
    print("\nPERFORMANCE TARGETS:")
    for target in targets_met:
        print(target)
    for target in targets_failed:
        print(target)
    
    # Save results to file
    with open("load_test_results.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDetailed results saved to: load_test_results.json")


if __name__ == "__main__":
    # Run load test programmatically
    import subprocess
    import sys
    
    print("Starting Financial RAG Load Test...")
    print("Target: 200 concurrent users for 10 minutes")
    print("Expected: <2s response time, >70% cache hit ratio")
    print("-" * 50)
    
    # Run locust command
    cmd = [
        sys.executable, "-m", "locust",
        "-f", __file__,
        "--host", "http://localhost:8000",
        "--users", "200",
        "--spawn-rate", "10",
        "--run-time", "10m",
        "--headless",
        "--print-stats",
        "--html", "load_test_report.html"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Load test failed: {e}")
        sys.exit(1) 