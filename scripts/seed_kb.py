"""
Seed initial knowledge base documents into NeonDB pgvector.
"""

import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.kb_manager import KnowledgeBaseManager

def main():
    kb = KnowledgeBaseManager()
    files = [
        ("technical_support_guide.md", "Technical Support Knowledge Guide", "Technical Support"),
        ("billing_policy.md", "Billing & Payment Policies", "Billing"),
        ("returns_and_refunds.md", "Returns, Refunds & Replacement Policies", "Returns"),
        ("general_inquiry_faq.md", "General Inquiry & Company Information", "General Inquiry"),
        ("complaints_escalation.md", "Customer Complaints & Dispute Redressal", "Complaint")
    ]

    for fname, title, cat in files:
        path = os.path.join("knowledge_base", fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            res = kb.ingest_document(
                title=title,
                content=content,
                category=cat,
                source=f"knowledge_base/{fname}",
                version="1.0"
            )
            print(f"Ingested {fname}: doc_id={res['document_id']}, chunks={res['chunks_indexed']}, status={res['status']}")

if __name__ == "__main__":
    main()

