import multiprocessing

# Gunicorn configuration for Render deployment
bind = "0.0.0.0:10000"
workers = 1  # Free tier: use minimal workers
worker_class = "sync"
timeout = 300  # 5 minutes for model loading
graceful_timeout = 120
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Memory optimization
max_requests = 100
max_requests_jitter = 20
preload_app = False  # Don't preload to save memory
