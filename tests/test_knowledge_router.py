import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from starlette.datastructures import UploadFile

from models.knowledge_base import DocumentMeta, KnowledgeBase
from routers.knowledge import DocumentResponse, KBCreateRequest, KBResponse, create_kb, list_kbs, upload_document


class TestKnowledgeRouter(unittest.TestCase):
    def test_kb_response_accepts_integer_id(self):
        resp = KBResponse(id=10, name="KB", description="desc")
        self.assertEqual(resp.id, 10)

    def test_document_response_accepts_integer_id(self):
        resp = DocumentResponse(id=20, file_name="a.txt", status="indexed")
        self.assertEqual(resp.id, 20)

    def test_list_kbs_returns_integer_ids(self):
        db = MagicMock()
        db.query.return_value.all.return_value = [SimpleNamespace(id=10, name="KB", description="desc")]

        result = list_kbs(db)

        self.assertEqual(result[0].id, 10)
        self.assertEqual(result[0].name, "KB")

    def test_create_kb_returns_integer_id(self):
        db = MagicMock()
        req = KBCreateRequest(name="KB", description="desc")

        def refresh_side_effect(kb):
            kb.id = 10
            kb.name = "KB"
            kb.description = "desc"

        db.refresh.side_effect = refresh_side_effect

        result = create_kb(req, db)

        self.assertEqual(result.id, 10)
        self.assertEqual(result.name, "KB")
        self.assertEqual(result.description, "desc")

    @patch("routers.knowledge.rag_service")
    @patch("routers.knowledge.process_document")
    def test_upload_document_returns_integer_id_and_string_metadata_doc_id(self, mock_process_document, mock_rag_service):
        mock_process_document.return_value = ["chunk-1", "chunk-2"]
        db = MagicMock()
        kb = SimpleNamespace(id=10)
        db.query.return_value.filter.return_value.first.return_value = kb

        async def run_test():
            upload = UploadFile(filename="hello.txt", file=BytesIO(b"hello"))

            def commit_side_effect():
                saved_doc = db.add.call_args.args[0]
                saved_doc.id = 20
                saved_doc.file_name = "hello.txt"
                saved_doc.status = "indexed"

            db.commit.side_effect = commit_side_effect

            result = await upload_document(file=upload, kb_id=10, db=db)

            metadata = mock_rag_service.add_texts.call_args.kwargs["metadatas"]
            self.assertEqual(len(metadata), 2)
            self.assertIsInstance(metadata[0]["doc_id"], str)
            self.assertNotEqual(metadata[0]["doc_id"], "")
            self.assertEqual(result.id, 20)
            self.assertEqual(result.file_name, "hello.txt")
            self.assertEqual(result.status, "indexed")

        import asyncio
        asyncio.run(run_test())


class TestKnowledgeModelBigintContract(unittest.TestCase):
    def test_knowledge_base_id_is_integer_column(self):
        self.assertEqual(KnowledgeBase.__table__.columns["id"].type.python_type, int)

    def test_document_meta_id_is_integer_column(self):
        self.assertEqual(DocumentMeta.__table__.columns["id"].type.python_type, int)

    def test_document_meta_kb_id_is_integer_column(self):
        self.assertEqual(DocumentMeta.__table__.columns["kb_id"].type.python_type, int)


if __name__ == "__main__":
    unittest.main()
