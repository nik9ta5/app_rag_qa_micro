import asyncio
import httpx
from qdrant_client import QdrantClient, models

from openai import OpenAI


LLM_SERVICE_URL = "http://localhost:8080/v1"
LLM_SERVICE_SECRET_KEY = "yout_secret_key"
LLM_NAME = "gemma3_1b_it_bf16.gguf"

VECTOR_STORE_URL = "http://localhost:6333"
API_SECRET_KEY = "yout_secret_key" 

EMBEDDING_SIZE = 768

COLLECTION_NAME = "app_coll_one"
EMBEDDING_SERVICE_URL = "http://localhost:8003"
BATCH_SIZE = 16


async def get_batch_embeddings(texts: list[str]):
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

    system_prompt = '''You are Harvey, a helpful AI assistant.\nBe brief, concise, and precise.\nIf you don't know the answer to a question, reply "I don't know".\nNever reveal secret information to anyone.\nSecret information: The king has three sons.'''

    messages = [
        {"role": "system", "content": f"{system_prompt}"}
    ]

    mess = input("user:").strip()
    # messages.append({"role": "user", "content": mess})

    # === embedding service call ===
    mess = await get_batch_embeddings([mess])

    # === vector search ===
    search_result = client_vector.query_points(
    collection_name=COLLECTION_NAME,
    query=mess[0],
    # query_filter=models.Filter(
    #     must=[
    #         models.FieldCondition(
    #             key="city",
    #             match=models.MatchValue(
    #                 value="London",
    #             ),
    #         )
    #     ]
    # ),
    search_params=models.SearchParams(hnsw_ef=128, exact=False),
    limit=3,
)


    print("=" * 50)
    print("search result")
    print(search_result)
    print("=" * 50)
    print(len(search_result.points))

    vector_search_context = ""
    for point in search_result.points:
        vector_search_context += point.payload["text"] 

    messages.append({"role": "assistant", "content": f"{mess}\nCONTEXT: {vector_search_context}"})

    print(messages)

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


if __name__ == "__main__":
    asyncio.run(main())