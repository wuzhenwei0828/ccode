import os
import tempfile
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter


# Extension -> loader function mapping
_LOADER_MAP: dict[str, str] = {
    ".pdf": "pypdf",
    ".docx": "docx",
    ".doc": "docx",
    ".txt": "text",
    ".md": "text",
    ".markdown": "text",
}


def load_document(file_path: str) -> str:
    """Load and extract text from a document file.

    Args:
        file_path: Path to the document file.

    Returns:
        Extracted text content.

    Raises:
        ValueError: If file format is not supported.
    """
    ext = Path(file_path).suffix.lower()
    loader_type = _LOADER_MAP.get(ext)

    if loader_type is None:
        supported = ", ".join(_LOADER_MAP.keys())
        raise ValueError(f"Unsupported file format '{ext}'. Supported: {supported}")

    if loader_type == "pypdf":
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
    elif loader_type == "docx":
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file_path)
    elif loader_type == "text":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"Unknown loader type: {loader_type}")

    docs = loader.load()
    return "\n".join(doc.page_content for doc in docs)


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks for embedding.

    Args:
        text: Full text content.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return chunks


def process_document(file_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Full pipeline: load document and split into chunks.

    Args:
        file_path: Path to the document file.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of text chunks ready for embedding.
    """
    text = load_document(file_path)
    if not text.strip():
        return []
    return chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
