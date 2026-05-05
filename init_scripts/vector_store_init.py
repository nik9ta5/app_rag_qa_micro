from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


# === to config ===
VECTOR_STORE_URL = "http://localhost:6333"
API_SECRET_KEY = "yout_secret_key"

COLLECTION_NAME = "app_coll_one"
EMBEDDING_SIZE = 768


if __name__ == "__main__":

    client = QdrantClient(
        url=VECTOR_STORE_URL,  
        api_key=API_SECRET_KEY,  
        timeout=60
    )

    # === init collection ===
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_SIZE,          
            distance=Distance.COSINE     
        )
    )

    if client.collection_exists(COLLECTION_NAME):
        print("collection create")

    print("done")