import datetime as dt
import asyncio
import httpx
from qdrant_client import QdrantClient, models

from openai import OpenAI

from utils.custom_logger import create_logger


LLM_SERVICE_URL = "http://localhost:8080/v1"
LLM_SERVICE_SECRET_KEY = "yout_secret_key"
LLM_NAME = "gemma3_1b_it_bf16.gguf"

VECTOR_STORE_URL = "http://localhost:6333"
API_SECRET_KEY = "yout_secret_key" 

EMBEDDING_SIZE = 768

COLLECTION_NAME = "app_coll_one"
EMBEDDING_SERVICE_URL = "http://localhost:8003"
BATCH_SIZE = 16

SYSTEM_PROMPT = '''You are a precise AI assistant.
Answer the question using ONLY the provided <context>.
The user's question is in the <question> block.
If <context> contains something related to the user's question in <question>, answer with a fragment from <context>.
Don't use extraneous knowledge.
If the answer doesn't match the context, answer precisely: "I don't know."'''


async def get_batch_embeddings(texts: list[str]):
    """Function for send text on embedding service"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{EMBEDDING_SERVICE_URL}/embed_batch",
                json={"texts": texts},
                timeout=120.0
            )
            response.raise_for_status()
            data = response.json()
            
            if 'embeddings' not in data:
                raise ValueError(f"Ответ от сервиса не содержит ключ 'embeddings': {data}")
                
            embeddings = data['embeddings']
            
            if any(e is None for e in embeddings):
                print("Warning: Some embeddings are None!")
                
            return embeddings
        
        except Exception as e:
            print(f"CRITICAL Error getting embeddings: {e}")
            raise e


async def main():
    print("RAG Core")

    app_logger = create_logger('../logs', "app_log.log")

    # === vector store datebase client init ===
    client_vector = QdrantClient(
        url=VECTOR_STORE_URL,  
        api_key=API_SECRET_KEY,  
        timeout=60
    )

    # === llm client init ===
    client = OpenAI(
        base_url=LLM_SERVICE_URL,
        api_key=LLM_SERVICE_SECRET_KEY
    )
    # ==========================================

    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}"}
    ]

    user_message = input("user:").strip()

    # === embedding service call ===
    mess = await get_batch_embeddings([user_message])

    # === vector search ===
    search_result = client_vector.query_points(
        collection_name=COLLECTION_NAME,
        query=mess[0],
        search_params=models.SearchParams(hnsw_ef=128, exact=False),
        limit=3,
    )

    app_logger.info(f"=== VECTOR SEARCH ===\n{search_result}\n\n{len(search_result.points)}")

    vector_search_context = ""
    for point in search_result.points:
        vector_search_context += point.payload["text"] + "\n"

    messages.append({"role": "user", "content": f"<context>\n{vector_search_context}</context>\n<question>{user_message}</question>"})


    # === log all messages ===
    total_char_len = 0
    for item in messages:
        app_logger.info(f"{item['role']}: {item['content']}")
        total_char_len += len(item['content'])

    app_logger.info(f"total_char_len: {total_char_len}\n=== END ===")

    # === stream answer generate ===
    stream = client.chat.completions.create(
        model=LLM_NAME,
        messages=messages,
        temperature=0.1,
        max_tokens=64,
        top_p=0.95,
        stream=True,
        stop=["<end_of_turn>", "<start_of_turn>"]
    )

    full_response = ""
    
    print("assistant: ", end="", flush=True)
    
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            text = chunk.choices[0].delta.content
            full_response += text
            print(text, end="", flush=True)
    print('\n')

    messages.append({"role": "assistant", "content": full_response})

    app_logger.info(f"\n=== MODEL ANSWER ===\n{full_response}\n")

if __name__ == "__main__":
    asyncio.run(main())