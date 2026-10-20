FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/papertrail/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system papertrail \
    && useradd --system --gid papertrail --create-home papertrail \
    && mkdir -p /app "$HF_HOME" \
    && chown -R papertrail:papertrail /app /home/papertrail

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
    && python -m pip install -r requirements.txt

COPY --chown=papertrail:papertrail . .
USER papertrail

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
