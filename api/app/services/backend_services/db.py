from typing import Optional
import threading
from dataclasses import dataclass, field
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from ...config import MONGODB_URI, DB_NAME


@dataclass
class ThreadLocalState:
    client: Optional[AsyncIOMotorClient] = field(default=None)
    database: Optional[AsyncIOMotorDatabase] = field(default=None)


class DatabaseManager:
    
    def __init__(self) -> None:
        self._local = threading.local()
    
    @property
    def state(self) -> ThreadLocalState:
        if not hasattr(self._local, "state"):
            self._local.state = ThreadLocalState()
        return self._local.state
    
    @property
    def client(self) -> Optional[AsyncIOMotorClient]:
        return self.state.client
    
    @property
    def database(self) -> Optional[AsyncIOMotorDatabase]:
        return self.state.database
    
    def get_database(self) -> AsyncIOMotorDatabase:
        if self.state.client is None:
            self.state.client = AsyncIOMotorClient(MONGODB_URI)
            # Explicitly specify database name to avoid "No default database defined" error
            self.state.database = self.state.client[DB_NAME]
        return self.state.database
    
    def get_client(self) -> AsyncIOMotorClient:
        if self.state.client is None:
            self.get_database()
        return self.state.client
    
    async def close(self) -> None:
        if self.state.client is not None:
            self.state.client.close()
            self.state.client = None
            self.state.database = None


_db = DatabaseManager()


def get_db() -> AsyncIOMotorDatabase:
    return _db.get_database()


def get_client() -> AsyncIOMotorClient:
    return _db.get_client()


async def close_db() -> None:
    await _db.close() 
