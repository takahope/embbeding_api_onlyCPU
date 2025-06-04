from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import uvicorn
from datetime import datetime
import logging
import asyncio
from functools import lru_cache
import os
import gc
import psutil
import signal
import sys
from contextlib import asynccontextmanager
from transformers import AutoTokenizer

# 設置日志 - 使用Python內置logging，輸出到控制台
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 支持的模型配置
SUPPORTED_MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "e5-large-v2": "intfloat/e5-large-v2", 
    "multilingual-e5-large": "intfloat/multilingual-e5-large",
    "bge-large-zh": "BAAI/bge-large-zh-v1.5",
    "text2vec-large": "shibing624/text2vec-large-chinese",
    "bge-large-en": "BAAI/bge-large-en-v1.5"
}

# 模型最大長度配置 - 保守設置以避免問題
MODEL_MAX_LENGTHS = {
    "bge-m3": 8192,
    "e5-large-v2": 512,
    "multilingual-e5-large": 512,
    "bge-large-zh": 512,
    "text2vec-large": 512,
    "bge-large-en": 512
}

# 安全的分塊大小 - 比最大長度小一些以留出安全邊界
SAFE_CHUNK_SIZES = {
    "bge-m3": 7500,  # 比 8192 小一些
    "e5-large-v2": 450,
    "multilingual-e5-large": 450,
    "bge-large-zh": 450,
    "text2vec-large": 450,
    "bge-large-en": 450
}

# 從環境變量獲取配置
NEW_MAX_LENGTH = int(os.getenv("NEW_MAX_LENGTH", 8192))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", 2))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 10))
MAX_TEXTS_PER_REQUEST = int(os.getenv("MAX_TEXTS_PER_REQUEST", 50))

# 全局模型緩存
model_cache: Dict[str, SentenceTransformer] = {}
tokenizer_cache: Dict[str, AutoTokenizer] = {}

# 創建信號量來限制並發請求數
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# OpenAI API 兼容的請求模型
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "bge-m3"
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"

class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

def get_memory_info():
    """獲取內存使用信息"""
    process = psutil.Process()
    return {
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "memory_percent": process.memory_percent(),
        "system_memory_mb": psutil.virtual_memory().available / 1024 / 1024
    }

def cleanup_memory():
    """清理內存 - 移除GPU相關代碼"""
    gc.collect()
    # 移除了torch.cuda.empty_cache()調用

def signal_handler(signum, frame):
    """信號處理器，優雅關閉"""
    logger.info(f"收到信號 {signum}，正在關閉...")
    cleanup_memory()
    sys.exit(0)

# 註冊信號處理器
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def count_tokens_safe(text: str, tokenizer) -> int:
    """安全的 token 計數函數，避免超長文本導致錯誤"""
    try:
        # 先檢查文本長度，如果太長就先截斷來計數
        if len(text) > 50000:  # 如果字符數超過50k，先粗略估算
            estimated_tokens = int(len(text.split()) * 1.3)
            if estimated_tokens > 20000:  # 如果估算超過20k token，直接返回估算值
                logger.warning(f"文本過長 ({len(text)} 字符)，使用估算token數: {estimated_tokens}")
                return estimated_tokens
        
        # 對於正常長度的文本，使用精確計數
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        return len(tokens)
    except Exception as e:
        logger.warning(f"Token計數失败，使用估算: {str(e)}")
        # 簡單估算：平均每個詞約1.3个token
        return int(len(text.split()) * 1.3)

def safe_chunked_encoding(text: str, model: SentenceTransformer, tokenizer, model_name: str) -> np.ndarray:
    """安全的分塊編碼函數 - 在tokenizer階段就進行分塊"""
    try:
        # 取得模型最大長度
        max_model_length = MODEL_MAX_LENGTHS.get(model_name, 8192)
        # 分塊大小不能超過模型最大長度
        safe_chunk_size = min(SAFE_CHUNK_SIZES.get(model_name, 7500), max_model_length)
        
        # 首先檢查token數量
        token_count = count_tokens_safe(text, tokenizer)
        
        # 僅當 token 數同時小於安全分塊與模型最大長度時才直接處理
        if token_count <= safe_chunk_size and token_count <= max_model_length:
            logger.debug(f"文本長度 {token_count} tokens，直接處理")
            return model.encode([text])[0]
        # 若在模型最大長度內但超過安全分塊，也允許直接處理
        elif token_count <= max_model_length:
            logger.debug(f"文本長度 {token_count} tokens，在模型最大長度內，直接處理")
            return model.encode([text])[0]
        
        logger.info(f"文本長度 {token_count} tokens 超過安全限制 {safe_chunk_size}，進行分塊處理")
        
        # 對文本進行tokenize
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        
        # 再次保證分塊大小不超過模型最大長度
        safe_chunk_size = min(safe_chunk_size, max_model_length)
        
        if len(tokens) <= safe_chunk_size:
            # 如果實際token數在安全範圍內，直接處理
            return model.encode([text])[0]
        
        # 分塊處理
        chunks = []
        overlap = min(128, safe_chunk_size // 10)  # 重疊大小，最多為分塊大小的10%
        
        for i in range(0, len(tokens), safe_chunk_size - overlap):
            chunk_tokens = tokens[i:i + safe_chunk_size]
            if len(chunk_tokens) == 0:
                continue
                
            try:
                chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                if chunk_text.strip():  # 只處理非空塊
                    chunks.append(chunk_text)
            except Exception as e:
                logger.warning(f"解碼分塊時出錯: {str(e)}")
                continue
        
        if not chunks:
            logger.error("分塊後沒有有效內容，回退到截斷處理")
            # 回退：直接截斷到安全長度
            truncated_tokens = tokens[:safe_chunk_size]
            truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            return model.encode([truncated_text])[0]
        
        logger.info(f"分塊完成，共 {len(chunks)} 個塊")
        
        # 批量編碼所有塊 - 使用小批次以避免內存問題
        all_embeddings = []
        chunk_batch_size = min(5, len(chunks))  # 每次最多處理5個塊
        
        for i in range(0, len(chunks), chunk_batch_size):
            batch_chunks = chunks[i:i + chunk_batch_size]
            try:
                batch_embeddings = model.encode(batch_chunks)
                all_embeddings.extend(batch_embeddings)
                
                # 定期清理內存
                if i > 0 and i % 10 == 0:
                    cleanup_memory()
                    
            except Exception as e:
                logger.error(f"處理分塊批次時出錯: {str(e)}")
                # 如果批次處理失败，嘗試逐個處理
                for chunk in batch_chunks:
                    try:
                        chunk_embedding = model.encode([chunk])[0]
                        all_embeddings.append(chunk_embedding)
                    except Exception as chunk_e:
                        logger.error(f"處理單個分塊時出錯: {str(chunk_e)}")
                        continue
        
        if not all_embeddings:
            raise Exception("所有分塊處理都失敗了")
        
        # 計算加權平均嵌入（根據塊的長度）
        embeddings_array = np.array(all_embeddings)
        
        # 簡單平均
        avg_embedding = np.mean(embeddings_array, axis=0)
        
        # 標準化
        norm = np.linalg.norm(avg_embedding)
        if norm > 1e-8:
            normalized_embedding = avg_embedding / norm
            return normalized_embedding
        else:
            logger.warning("平均嵌入的範數過小")
            return avg_embedding
            
    except Exception as e:
        logger.error(f"分塊編碼時發生嚴重錯誤: {str(e)}")
        # 最終回退：截斷處理
        try:
            max_model_length = MODEL_MAX_LENGTHS.get(model_name, 8192)
            safe_chunk_size = min(SAFE_CHUNK_SIZES.get(model_name, 7500), max_model_length)
            tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
            truncated_tokens = tokens[:safe_chunk_size]
            truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            logger.info(f"回退到截斷處理，保留 {len(truncated_tokens)} tokens")
            return model.encode([truncated_text])[0]
        except Exception as fallback_e:
            logger.error(f"回退處理也失敗: {str(fallback_e)}")
            raise HTTPException(status_code=500, detail="文本處理失敗，請嘗試較短的文本")

def load_model_and_tokenizer(model_name: str) -> tuple[SentenceTransformer, AutoTokenizer]:
    """加載並緩存模型和分詞器 - 強制使用CPU"""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"不支持的模型: {model_name}")
    
    model = None
    tokenizer = None
    
    # 加載模型
    if model_name in model_cache:
        model = model_cache[model_name]
    else:
        logger.info(f"正在加載模型: {model_name}")
        memory_before = get_memory_info()
        logger.info(f"加載前內存使用: {memory_before['memory_mb']:.1f}MB")
        
        try:
            model_path = SUPPORTED_MODELS[model_name]
            
            # 強制使用CPU設備
            device = "cpu"
            
            # 加載模型時設置參數以減少內存使用
            model = SentenceTransformer(
                model_path, 
                trust_remote_code=True,
                device=device
            )
            
            # 設置CPU線程數
            import torch
            torch.set_num_threads(2)
            
            model_cache[model_name] = model
            
            memory_after = get_memory_info()
            logger.info(f"模型 {model_name} 加載成功，使用設備: {device}")
            logger.info(f"加載後內存使用: {memory_after['memory_mb']:.1f}MB (+{memory_after['memory_mb'] - memory_before['memory_mb']:.1f}MB)")
            
        except Exception as e:
            logger.error(f"加載模型 {model_name} 失敗: {str(e)}")
            cleanup_memory()
            raise HTTPException(status_code=500, detail=f"模型加載失敗: {str(e)}")
    
    # 加載分詞器
    if model_name in tokenizer_cache:
        tokenizer = tokenizer_cache[model_name]
    else:
        logger.info(f"正在加載分詞器: {model_name}")
        try:
            model_path = SUPPORTED_MODELS[model_name]
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            tokenizer_cache[model_name] = tokenizer
            logger.info(f"分詞器 {model_name} 加載成功")
        except Exception as e:
            logger.error(f"加載分詞器 {model_name} 失敗: {str(e)}")
            # 如果分詞器加載失敗，使用None，後續會使用估算方法
            tokenizer = None
    
    return model, tokenizer

def preprocess_text(text: str, model_name: str) -> str:
    """根據模型類型預處理文本"""
    if model_name.startswith("e5"):
        if not text.startswith("query:") and not text.startswith("passage:"):
            return f"query: {text}"
    return text

def postprocess_embeddings(embeddings: np.ndarray, dimensions: Optional[int] = None) -> np.ndarray:
    """後處理嵌入向量"""
    # 標準化
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # 降維（如果指定）
    if dimensions and dimensions < embeddings.shape[1]:
        embeddings = embeddings[:, :dimensions]
    
    return embeddings

async def process_embeddings_in_batches_async(model: SentenceTransformer, texts: List[str], 
                                            batch_size: int = MAX_BATCH_SIZE, request_id: str = "", 
                                            tokenizer=None, model_name: str = "") -> np.ndarray:
    """異步分批處理嵌入 - 改進的安全版本"""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        logger.info(f"[請求{request_id}] 處理批次 {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}，大小: {len(batch_texts)}")
        
        try:
            # 使用異步方式處理當前批次
            def encode_batch():
                batch_embeddings = []
                for text in batch_texts:
                    try:
                        if tokenizer:
                            # 使用安全的分塊編碼
                            embedding = safe_chunked_encoding(text, model, tokenizer, model_name)
                        else:
                            # 沒有分詞器時，嘗試直接編碼，但限制文本長度
                            if len(text) > 10000:  # 如果文本很長，先截斷
                                logger.warning(f"文本過長且無分詞器，截斷到10000字符")
                                text = text[:10000]
                            embedding = model.encode([text])[0]
                        
                        batch_embeddings.append(embedding)
                        
                    except Exception as text_e:
                        logger.error(f"處理單個文本時出錯: {str(text_e)}")
                        # 嘗試截斷文本後重試
                        try:
                            truncated_text = text[:1000] if len(text) > 1000 else text
                            embedding = model.encode([truncated_text])[0]
                            batch_embeddings.append(embedding)
                            logger.info(f"截斷處理成功")
                        except Exception as retry_e:
                            logger.error(f"截斷重試也失敗: {str(retry_e)}")
                            raise HTTPException(status_code=500, detail=f"文本處理失敗: {str(text_e)}")
                
                return np.array(batch_embeddings)
            
            # 在線程池中執行編碼
            loop = asyncio.get_event_loop()
            batch_embeddings = await loop.run_in_executor(None, encode_batch)
            all_embeddings.append(batch_embeddings)
            
            # 清理內存
            if i > 0 and i % (batch_size * 2) == 0:
                cleanup_memory()
                memory_info = get_memory_info()
                logger.info(f"[請求{request_id}] 批次處理後內存使用: {memory_info['memory_mb']:.1f}MB")
            
            # 讓出控制權
            await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"[請求{request_id}] 處理批次時發生錯誤: {str(e)}")
            cleanup_memory()
            raise
    
    # 合併所有嵌入
    if all_embeddings:
        result = np.vstack(all_embeddings)
        cleanup_memory()
        return result
    else:
        return np.array([])

# 生命週期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動
    logger.info("Enhanced Embedding API Server (CPU-only) 啟動中...")
    logger.info(f"運行模式: CPU Only")
    logger.info(f"最大並發請求數: {MAX_CONCURRENT_REQUESTS}")
    logger.info(f"批次大小: {MAX_BATCH_SIZE}")
    logger.info(f"安全分塊大小: {SAFE_CHUNK_SIZES}")
    memory_info = get_memory_info()
    logger.info(f"啟動時內存: {memory_info['memory_mb']:.1f}MB (可用: {memory_info['system_memory_mb']:.1f}MB)")
    
    yield
    
    # 關閉
    logger.info("正在關閉服務器...")
    cleanup_memory()
    logger.info("服務器已關閉")

app = FastAPI(
    title="Enhanced OpenAI Compatible Embedding API (CPU-only)", 
    version="1.2.0-cpu",
    lifespan=lifespan
)

@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """列出所有可用模型"""
    models = [
        ModelInfo(
            id=model_name,
            created=int(datetime.now().timestamp())
        )
        for model_name in SUPPORTED_MODELS.keys()
    ]
    
    return ModelsResponse(data=models)

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """創建文本嵌入 - 改進的CPU版本，強化長文本處理"""
    # 生成請求ID用於日志追蹤
    request_id = f"{datetime.now().strftime('%H%M%S')}-{id(request) % 1000:03d}"
    
    # 使用信號量限制並發請求數
    async with request_semaphore:
        try:
            # 記錄請求開始時的內存狀態
            memory_start = get_memory_info()
            logger.info(f"[請求{request_id}] 開始處理，當前內存: {memory_start['memory_mb']:.1f}MB")
            logger.info(f"[請求{request_id}] 當前活躍請求數: {MAX_CONCURRENT_REQUESTS - request_semaphore._value}")
            
            # 驗證模型
            if request.model not in SUPPORTED_MODELS:
                raise HTTPException(
                    status_code=400,
                    detail=f"不支持的模型: {request.model}. 支持的模型: {list(SUPPORTED_MODELS.keys())}"
                )
            
            # 處理輸入
            if isinstance(request.input, str):
                texts = [request.input]
            else:
                texts = request.input
            
            if not texts:
                raise HTTPException(status_code=400, detail="輸入不能為空")
            
            # 限制請求大小
            if len(texts) > MAX_TEXTS_PER_REQUEST:
                raise HTTPException(
                    status_code=400, 
                    detail=f"文本數量超過限制 ({len(texts)} > {MAX_TEXTS_PER_REQUEST})"
                )
            
            # 預處理文本
            processed_texts = [preprocess_text(text, request.model) for text in texts]
            
            # 加載模型和分詞器
            model, tokenizer = load_model_and_tokenizer(request.model)
            
            # 生成嵌入
            logger.info(f"[請求{request_id}] 為 {len(processed_texts)} 個文本生成嵌入，使用模型: {request.model} (CPU)")
            
            # 使用改進的異步分批處理
            embeddings = await process_embeddings_in_batches_async(
                model, processed_texts, MAX_BATCH_SIZE, request_id, tokenizer, request.model
            )
            
            # 後處理
            embeddings = postprocess_embeddings(embeddings, request.dimensions)
            
            # 構造響應
            data = []
            for i, embedding in enumerate(embeddings):
                data.append({
                    "object": "embedding",
                    "index": i,
                    "embedding": embedding.tolist()
                })
            
            # 計算token數量（使用安全方法）
            if tokenizer:
                total_tokens = sum(count_tokens_safe(text, tokenizer) for text in texts)
                logger.info(f"[請求{request_id}] token計數: {total_tokens}")
            else:
                # 回退到估算方法
                total_tokens = sum(len(text.split()) for text in texts)
                logger.info(f"[請求{request_id}] 估算token計數: {total_tokens}")
            
            # 記錄內存使用
            memory_end = get_memory_info()
            logger.info(f"[請求{request_id}] 處理完成，內存: {memory_end['memory_mb']:.1f}MB (變化: {memory_end['memory_mb'] - memory_start['memory_mb']:+.1f}MB)")
            
            # 清理內存
            cleanup_memory()
            
            return EmbeddingResponse(
                data=data,
                model=request.model,
                usage={
                    "prompt_tokens": total_tokens,
                    "total_tokens": total_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"[請求{request_id}] 生成嵌入時發生錯誤: {str(e)}")
            cleanup_memory()
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康檢查"""
    memory_info = get_memory_info()
    active_requests = MAX_CONCURRENT_REQUESTS - request_semaphore._value
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "device": "cpu",
        "memory_mb": memory_info["memory_mb"],
        "memory_percent": memory_info["memory_percent"],
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "active_requests": active_requests,
        "available_slots": request_semaphore._value,
        "batch_size": MAX_BATCH_SIZE,
        "safe_chunk_sizes": SAFE_CHUNK_SIZES
    }

@app.get("/")
async def root():
    """根路徑"""
    return {
        "message": "Enhanced OpenAI Compatible Embedding API Server (CPU-only)",
        "version": "1.2.0-cpu",
        "device": "cpu",
        "supported_models": list(SUPPORTED_MODELS.keys()),
        "safe_chunk_sizes": SAFE_CHUNK_SIZES,
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "batch_size": MAX_BATCH_SIZE,
        "features": [
            "CPU-only processing",
            "Safe long text processing with chunking",
            "Memory-optimized token counting",
            "Concurrent request handling",
            "Advanced memory management",
            "OpenAI API compatibility",
            "Robust error handling"
        ],
        "endpoints": {
            "models": "/v1/models",
            "embeddings": "/v1/embeddings",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    # 配置服務器參數
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8090))
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info",
        access_log=True,
        workers=1  # 單一工作進程避免內存問題
    )