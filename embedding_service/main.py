import asyncio
import uvicorn
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer

import torch


PORT = 8000
app = FastAPI(title="Embedding Service")


device = "cuda" if torch.cuda.is_available() else "cpu"


model = SentenceTransformer(
    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2', 
    cache_folder = "./embedding_models",
    device=device
)


model_executor = ThreadPoolExecutor(max_workers=1)


# === schemes ===

class TextRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: list[str]

class EmbeddingResponse(BaseModel):
    embedding: list[float]

# Схема для батча эмбеддингов (список списков)
class BatchEmbeddingResponse(BaseModel):
    embeddings: list[list[float]]




def compute_embeddings(texts: list[str]):
    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )
    return embeddings.tolist()




@app.post("/embed", response_model=EmbeddingResponse)
async def embed_text(request: TextRequest):
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(model_executor, compute_embeddings, [request.text])
        return {"embedding": result[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed_batch", response_model=BatchEmbeddingResponse)
async def embed_batch(request: BatchRequest):
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(model_executor, compute_embeddings, request.texts)
        return {"embeddings": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)