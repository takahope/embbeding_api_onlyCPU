# 使用 Python 3.10 作為基礎鏡像
FROM python:3.10-slim

# 設置工作目錄
WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    procps \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 復制 requirements.txt
COPY requirements.txt .

# 安裝 Python 依賴
# 首先安裝CPU版本的PyTorch
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# 復制應用代碼
COPY main.py .

# 創建必要目錄和設置權限
RUN mkdir -p /home/app/.cache/huggingface && \
    useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app && \
    chown -R app:app /home/app

# 切換到非 root 用戶
USER app

# 暴露端口
EXPOSE 8000

# 設置環境變量
ENV HOST=0.0.0.0
ENV PORT=8000
ENV PYTHONPATH=/app

# Python 優化設置
ENV PYTHONUNBUFFERED=1
ENV PYTHONOPTIMIZE=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TOKENIZERS_PARALLELISM=false

# CPU 線程優化
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
ENV NUMEXPR_NUM_THREADS=2
ENV OPENBLAS_NUM_THREADS=2

# 強制使用CPU - 禁用CUDA
ENV CUDA_VISIBLE_DEVICES=""

# 應用程序配置（與main.py中的環境變量對應）
ENV MAX_CONCURRENT_REQUESTS=3
ENV MAX_BATCH_SIZE=20
ENV MAX_TEXTS_PER_REQUEST=100
ENV NEW_MAX_LENGTH=8192

# HuggingFace 緩存配置
ENV HF_HOME=/home/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/app/.cache/huggingface/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/home/app/.cache/huggingface/sentence-transformers

# 健康檢查
HEALTHCHECK --interval=30s --timeout=20s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 啟動命令（與main.py中的uvicorn配置一致）
CMD ["python", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--access-log"]