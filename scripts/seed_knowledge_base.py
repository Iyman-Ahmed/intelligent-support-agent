"""
Seed the knowledge base with initial FAQ documents.

Usage:
    python scripts/seed_knowledge_base.py --source docs/
    python scripts/seed_knowledge_base.py --mock   # Seed with built-in mock data
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


async def seed_from_directory(docs_dir: str) -> None:
    """Load markdown/text files from a directory into the knowledge base."""
    from app.services.knowledge_base import KnowledgeBaseService
    from app.config import settings

    kb = KnowledgeBaseService(
        use_vector_db=bool(settings.pinecone_api_key or settings.database_url),
        pinecone_api_key=settings.pinecone_api_key,
        pinecone_index=settings.pinecone_index,
        db_url=settings.database_url,
    )
    await kb.initialize()

    count = 0
    for root, _, files in os.walk(docs_dir):
        for fname in files:
            if not fname.endswith((".md", ".txt", ".rst")):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, "r") as f:
                content = f.read().strip()
            if not content:
                continue
            doc_id = f"kb_{fname.replace('.', '_')}"
            title = fname.replace("-", " ").replace("_", " ").replace(".md", "").title()
            await kb.add_document(doc_id=doc_id, title=title, content=content)
            print(f"✅ Seeded: {title}")
            count += 1

    print(f"\n🎉 Seeded {count} documents into the knowledge base")


async def seed_mock_data() -> None:
    """Print the built-in mock KB entries."""
    from app.services.knowledge_base import MOCK_KB
    print(f"Built-in mock knowledge base has {len(MOCK_KB)} entries:")
    for doc in MOCK_KB:
        print(f"  [{doc['category']}] {doc['title']}")
    print("\nTo use vector search, configure PINECONE_API_KEY or DATABASE_URL in .env")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the knowledge base")
    parser.add_argument("--source", help="Directory containing docs to seed")
    parser.add_argument("--mock", action="store_true", help="Show built-in mock data")
    args = parser.parse_args()

    if args.mock:
        asyncio.run(seed_mock_data())
    elif args.source:
        asyncio.run(seed_from_directory(args.source))
    else:
        print("Usage: python scripts/seed_knowledge_base.py --mock")
        print("   or: python scripts/seed_knowledge_base.py --source docs/")
