"""Gunicorn configuration for Railway deployments."""

import os


bind = f"0.0.0.0:{os.environ.get('PORT', '5001')}"
workers = 1
worker_class = "gthread"
threads = int(os.environ.get("GUNICORN_THREADS", "8"))
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "300"))
graceful_timeout = int(os.environ.get("GUNICORN_GRACEFUL_TIMEOUT", "30"))
keepalive = int(os.environ.get("GUNICORN_KEEPALIVE", "5"))
loglevel = os.environ.get("GUNICORN_LOG_LEVEL", "info")
accesslog = "-"
errorlog = "-"
capture_output = True
preload_app = False
worker_tmp_dir = "/dev/shm"
