from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
from functools import wraps
import logging

# Define metrics
conversion_requests = Counter(
    'paper_conversion_requests_total',
    'Total number of paper conversion requests',
    ['status']
)

conversion_duration = Histogram(
    'paper_conversion_duration_seconds',
    'Time spent processing paper conversion',
    buckets=[10, 30, 60, 120, 300, 600, 1800]
)

active_jobs = Gauge(
    'paper_conversion_active_jobs',
    'Number of currently active conversion jobs'
)

paper_size = Histogram(
    'paper_size_bytes',
    'Size of processed papers in bytes',
    buckets=[1e5, 1e6, 5e6, 1e7, 5e7, 1e8]
)

equation_count = Histogram(
    'paper_equation_count',
    'Number of equations per paper',
    buckets=[0, 5, 10, 20, 50, 100]
)

table_count = Histogram(
    'paper_table_count',
    'Number of tables per paper',
    buckets=[0, 2, 5, 10, 20, 50]
)

audio_duration = Histogram(
    'audio_duration_seconds',
    'Duration of generated audio',
    buckets=[60, 300, 600, 1800, 3600]
)

tts_latency = Histogram(
    'tts_generation_latency_seconds',
    'Latency of TTS generation',
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
)

def monitor_conversion(func):
    """Decorator to monitor conversion functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            conversion_requests.labels(status='success').inc()
            return result
        except Exception as e:
            conversion_requests.labels(status='failed').inc()
            raise e
        finally:
            duration = time.time() - start_time
            conversion_duration.observe(duration)
    
    return wrapper

def track_active_jobs(func):
    """Decorator to track active jobs"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        active_jobs.inc()
        try:
            return func(*args, **kwargs)
        finally:
            active_jobs.dec()
    
    return wrapper

class MetricsMiddleware:
    """FastAPI middleware for request metrics"""
    
    async def __call__(self, request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        duration = time.time() - start_time
        
        # Record request duration
        conversion_duration.observe(duration)
        
        # Add metrics headers
        response.headers['X-Processing-Time'] = str(duration)
        
        return response