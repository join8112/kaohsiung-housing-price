# Container for the valuation API (api/main.py).  The trained model is baked
# into the image, so the container has no runtime dependency on any data store.
#
#   docker build -t khp-api .
#   docker run --rm -p 8000:8000 khp-api        # then open http://localhost:8000/docs
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Dependencies first, so code changes don't invalidate the pip layer.
COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY src/khp ./src/khp
COPY api ./api
COPY models ./models

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,os,sys; sys.exit(0 if urllib.request.urlopen(f\"http://127.0.0.1:{os.environ['PORT']}/health\").status==200 else 1)"

# Render (and most PaaS) inject $PORT; locally it defaults to 8000.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
