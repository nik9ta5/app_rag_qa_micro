import os

if __name__ == "__main__":
    print('create needs dirs')

    os.makedirs("../llm_service/llama-cache", exist_ok=True)
    os.makedirs("../llm_service/models", exist_ok=True)
    os.makedirs("../vector_store/qdrant_snapshots", exist_ok=True)
    os.makedirs("../vector_store/qdrant_storage", exist_ok=True)
    os.makedirs("../embedding_service/embedding_models", exist_ok=True)

    print("done")