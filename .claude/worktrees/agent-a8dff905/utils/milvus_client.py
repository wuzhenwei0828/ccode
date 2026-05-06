from langchain_milvus import Milvus

from config import get_settings
from services.embedding_service import get_embeddings


def get_milvus_store(collection_name: str | None = None) -> Milvus:
    """Get a Milvus vector store instance.

    Args:
        collection_name: Override the default collection name from config.

    Returns:
        LangChain Milvus vector store instance.
    """
    settings = get_settings()
    milvus_cfg = settings.get_milvus_config()
    embeddings = get_embeddings()

    uri = f"http://{milvus_cfg.host}:{milvus_cfg.port}"
    return Milvus(
        embedding_function=embeddings,
        collection_name=collection_name or milvus_cfg.collection_name,
        connection_args={"uri": uri},
    )
