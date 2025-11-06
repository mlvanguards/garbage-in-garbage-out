import logging
from typing import Optional

from qdrant_client import QdrantClient

from src.config import settings

logger = logging.getLogger(__name__)

class QDrantConnectionManager:

    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._client: Optional[QdrantClient] = None

    def init(self):
        try:
            self._client = QdrantClient(
                url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=60
            )

            logger.info("Connected to Qdrant.")
        except Exception as e:
            logger.exception(f"Could not connect to Qdrant: {e}")
            raise

    def get_client(self) -> QdrantClient:
        return self._client

    def close(self):
        if self._client:
            self._client.close()
            logger.info("Qdrant connection was closed.")

    def connected(self) -> bool:
        if not self._client:
            return False
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False


connection_manager = QDrantConnectionManager()

if __name__ == "__main__":
    connection_manager.init()
    print(connection_manager.connected())
    connection_manager.close()