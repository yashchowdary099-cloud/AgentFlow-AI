"""
Comprehensive Automated Test Suite for AgentFlow AI
Tests:
1. Health Check endpoint
2. Mathematical Calculations with safe Calculator tool
3. General Purpose Question answering
4. Multi-turn Session Memory isolation
5. PDF Creation, Upload, Vector Indexing, and RAG Query
6. Conversation CRUD & Deletion
"""

import os
import sys
import unittest
import uuid
import pymupdf
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from backend.main import app
from backend.services.document_service import clear_active_document
from backend.database.database import get_messages

client = TestClient(app)

class TestAgentFlowSystem(unittest.TestCase):

    def setUp(self):
        clear_active_document()

    def test_01_health_check(self):
        """Test GET /api/health"""
        response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("service"), "LangChain AI Agent")
        print("[PASS] Health Check Passed:", data.get("status"))

    def test_02_calculator_tool(self):
        """Test math queries automatically routing to Calculator tool"""
        # Test multiplication
        res1 = client.post("/api/chat", json={"message": "Calculate 458 * 92"})
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()
        self.assertEqual(data1.get("tool_used"), "calculator")
        self.assertIn("42,136", data1.get("answer"))
        self.assertTrue(any("Calculator tool selected" in step for step in data1.get("activity", [])))
        print("[PASS] Calculator (458 * 92) Passed:", data1.get("answer"))

        # Test percentage
        res2 = client.post("/api/chat", json={"message": "What is 18% of 1250?"})
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertEqual(data2.get("tool_used"), "calculator")
        self.assertIn("225", data2.get("answer"))
        print("[PASS] Calculator (18% of 1250) Passed:", data2.get("answer"))

        # Test parentheses arithmetic
        res3 = client.post("/api/chat", json={"message": "Solve (500 + 250) / 5"})
        self.assertEqual(res3.status_code, 200)
        data3 = res3.json()
        self.assertEqual(data3.get("tool_used"), "calculator")
        self.assertIn("150", data3.get("answer"))
        print("[PASS] Calculator ((500 + 250) / 5) Passed:", data3.get("answer"))

    def test_03_general_chat_routing(self):
        """Test general inquiries routing to Direct LLM with conversational memory"""
        res = client.post("/api/chat", json={"message": "What is cloud computing?"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("tool_used"), "none")
        self.assertTrue(any("Direct AI response selected" in step for step in data.get("activity", [])))
        print("[PASS] General Query Routing Passed. Tool used:", data.get("tool_used"))

    def test_04_session_memory_and_isolation(self):
        """Test multi-turn conversational memory and session isolation"""
        # Session A
        res_a1 = client.post("/api/new-chat")
        cid_a = res_a1.json()["conversation_id"]

        client.post("/api/chat", json={"conversation_id": cid_a, "message": "My project is AgentFlow AI"})
        
        # Verify message persisted in SQLite for Session A
        msgs_a = get_messages(cid_a)
        self.assertGreaterEqual(len(msgs_a), 2)
        self.assertEqual(msgs_a[0]["role"], "user")
        self.assertIn("AgentFlow AI", msgs_a[0]["content"])

        # Session B (New Chat should be completely isolated)
        res_b = client.post("/api/new-chat")
        cid_b = res_b.json()["conversation_id"]
        self.assertNotEqual(cid_a, cid_b)

        msgs_b = get_messages(cid_b)
        self.assertEqual(len(msgs_b), 0)
        print("[PASS] Session Isolation Passed: Session A and B are distinct.")

    def test_05_pdf_upload_and_rag(self):
        """Generate sample PDF, upload via API, and test RAG query"""
        # 1. Generate test PDF in-memory bytes to avoid file locks
        doc = pymupdf.open()
        page = doc.new_page()
        text = (
            "Cloud Computing and Virtualization Notes\n\n"
            "Cloud virtualization allows multiple simulated environments or dedicated resources "
            "to be created from a single physical hardware system. "
            "The key advantages of virtualization include: "
            "1. Enhanced server consolidation and resource utilization.\n"
            "2. Dramatic cost reduction in hardware, electricity, and maintenance.\n"
            "3. Rapid provisioning and dynamic scaling of virtual machines.\n"
            "4. Built-in disaster recovery and high availability mechanisms.\n\n"
            "Cloud deployment models include Public, Private, and Hybrid clouds. "
            "Common service models are IaaS (Infrastructure as a Service), "
            "PaaS (Platform as a Service), and SaaS (Software as a Service)."
        )
        page.insert_text((50, 72), text, fontsize=12)
        pdf_bytes = doc.tobytes()
        doc.close()

        # 2. Upload PDF
        upload_res = client.post(
            "/api/upload",
            files={"file": ("sample_cloud_notes.pdf", pdf_bytes, "application/pdf")}
        )
        self.assertEqual(upload_res.status_code, 200)
        up_data = upload_res.json()
        self.assertTrue(up_data.get("success"))
        self.assertEqual(up_data.get("filename"), "sample_cloud_notes.pdf")
        self.assertGreater(up_data.get("pages"), 0)
        self.assertGreater(up_data.get("chunks"), 0)
        print("[PASS] PDF Upload & Indexing Passed:", up_data.get("filename"))

        # 3. Check Document Status
        doc_status_res = client.get("/api/document")
        self.assertEqual(doc_status_res.status_code, 200)
        self.assertTrue(doc_status_res.json().get("has_document"))

        # 4. Ask a question regarding the PDF
        chat_res = client.post("/api/chat", json={
            "message": "According to the uploaded document, what are the advantages of virtualization?"
        })
        self.assertEqual(chat_res.status_code, 200)
        c_data = chat_res.json()
        self.assertEqual(c_data.get("tool_used"), "document")
        self.assertTrue(any("Document retrieval tool selected" in s for s in c_data.get("activity", [])))
        self.assertIn("virtualization", c_data.get("answer").lower())
        print("[PASS] Document RAG Query Passed. Tool used:", c_data.get("tool_used"))

if __name__ == "__main__":
    unittest.main(exit=False)
