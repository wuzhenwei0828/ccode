import os
import tempfile

import pytest

from services.document_loader import chunk_text, load_document, process_document


class TestChunkText:
    def test_chunk_text_splits_by_size(self):
        text = "a" * 1000
        chunks = chunk_text(text, chunk_size=300, chunk_overlap=0)
        assert len(chunks) >= 4
        assert all(len(c) <= 300 for c in chunks)

    def test_chunk_text_with_overlap(self):
        text = "a" * 500
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)
        assert len(chunks) >= 2
        for i in range(len(chunks) - 1):
            assert len(chunks[i]) > 0
            assert len(chunks[i + 1]) > 0

    def test_chunk_text_short_input(self):
        text = "short text"
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text


class TestLoadDocument:
    def test_load_text_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("hello world")
            f.flush()
            result = load_document(f.name)
        os.unlink(f.name)
        assert result == "hello world"

    def test_load_md_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("# Title\n\ncontent")
            f.flush()
            result = load_document(f.name)
        os.unlink(f.name)
        assert "# Title" in result

    def test_load_unsupported_format(self):
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_document("test.csv")
