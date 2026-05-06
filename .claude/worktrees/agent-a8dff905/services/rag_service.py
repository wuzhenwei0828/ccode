import logging
from typing import Optional

from langchain_core.documents import Document

from services.embedding_service import get_embeddings
from utils.milvus_client import get_milvus_store

logger = logging.getLogger(__name__)


class RAGService:
    """RAG pipeline: index documents into Milvus and query for retrieval."""

    def __init__(self):
        self._store = None

    @property
    def store(self):
        """Lazy-init the vector store."""
        if self._store is None:
            self._store = get_milvus_store()
        return self._store

    def add_texts(self, texts: list[str], metadatas: Optional[list[dict]] = None) -> list[str]:
        """Index text chunks into Milvus.

        Args:
            texts: List of text chunks to index.
            metadatas: Optional metadata per chunk (e.g., doc_id, kb_id, file_name).

        Returns:
            List of inserted document IDs.
        """
        ids = self.store.add_texts(texts, metadatas=metadatas)
        logger.info(f"Indexed {len(ids)} text chunks into Milvus")
        return ids

    def query(self, question: str, k: int = 4) -> list[Document]:
        """Retrieve relevant documents for a question.

        Args:
            question: User's query text.
            k: Number of documents to retrieve.

        Returns:
            List of relevant Document objects.
        """
        results = self.store.similarity_search(question, k=k)
        logger.info(f"Retrieved {len(results)} documents for query: {question[:50]}...")
        return results

    def delete_by_metadata(self, **metadata_filter) -> int:
        """Delete documents from Milvus by metadata filter.

        Args:
            **metadata_filter: Key-value pairs to match (e.g., doc_id="xxx").

        Returns:
            Number of deleted documents.
        """
        # Milvus delete via expression
        # Build filter expression from metadata
        conditions = []
        for key, value in metadata_filter.items():
            conditions.append(f'{key} == "{value}"')
        expr = " and ".join(conditions)
        result = self.store.delete(expr=expr)
        logger.info(f"Deleted documents matching {metadata_filter}: {result}")
        return result


# Singleton
rag_service = RAGService()
