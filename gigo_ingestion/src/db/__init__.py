"""
Database module for Qdrant vector database operations.

This module provides:
- QDrantConnectionManager: Singleton for managing Qdrant client connections
- QdrantCollection: Class for managing collections and embeddings
"""

from src.db.collection import QdrantCollection
from src.db.manager import QDrantConnectionManager, connection_manager

__all__ = [
    "QDrantConnectionManager",
    "QdrantCollection",
    "connection_manager",
]

