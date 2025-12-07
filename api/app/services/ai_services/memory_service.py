"""
Memory service for managing user-specific memories using Mem0.
Provides persistent memory storage across chat sessions.
"""
import logging
import asyncio
from functools import partial
from typing import List, Dict, Any, Optional
from mem0 import MemoryClient
from app.config import MEM0_API_KEY, MONGODB_URI, MEM0_STORAGE_TYPE
from app.services.backend_services.db import get_db

logger = logging.getLogger(__name__)

_memory_client = None


def get_memory_client() -> Optional[MemoryClient]:
    """Get or create Mem0 client instance (singleton)"""
    global _memory_client
    
    if _memory_client is not None:
        return _memory_client
    
    try:
        # Configure Mem0 based on whether API key is provided (hosted) or not (self-hosted)
        if MEM0_API_KEY:
            # Hosted Mem0 platform
            _memory_client = MemoryClient(api_key=MEM0_API_KEY)
            logger.info("✅ [MEMORY] Initialized Mem0 client with hosted platform")
        else:
            if not MONGODB_URI:
                raise RuntimeError("Mem0 self-hosted mode requires MONGODB_URI to be configured")
            config = {
                "storage": {
                    "type": MEM0_STORAGE_TYPE,
                    "connection_string": MONGODB_URI,
                }
            }
            _memory_client = MemoryClient(config=config)
            logger.info(f"✅ [MEMORY] Initialized Mem0 client with {MEM0_STORAGE_TYPE} storage")
        
        return _memory_client
    except Exception as e:
        logger.error(f"❌ [MEMORY] Failed to initialize Mem0 client: {e}", exc_info=True)
        raise


class MemoryService:
    """Service for managing user memories using Mem0"""
    
    def __init__(self):
        self.client = get_memory_client()
    
    def _ensure_client(self) -> bool:
        """Ensure client is available, return False if not"""
        if self.client is None:
            logger.warning("⚠️ [MEMORY] Mem0 client not available, memory operations disabled")
            return False
        return True
    
    async def get_memories(
        self, 
        user_email: str, 
        query: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories for a user based on current query.
        
        Args:
            user_email: User identifier (used as agent_id in Mem0)
            query: Current query to search memories for
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of memory dictionaries with 'memory' and 'metadata' fields
        """
        if not self._ensure_client():
            return []
        
        try:
            # Mem0 search is synchronous, run it in thread pool to avoid blocking
            # Use functools.partial to handle keyword arguments with asyncio.to_thread
            search_func = partial(
                self.client.search,
                query=query,
                user_id=user_email,
                limit=limit
            )
            memories = await asyncio.to_thread(search_func)
            
            logger.info(f"✅ [MEMORY] Retrieved {len(memories)} memories for user {user_email}")
            return memories
        except Exception as e:
            logger.error(f"❌ [MEMORY] Error retrieving memories for {user_email}: {e}", exc_info=True)
            return []
    
    async def add_memory(
        self, 
        user_email: str, 
        messages: List[Dict[str, str]]
    ) -> bool:
        """
        Store conversation memories after each chat interaction.
        
        Args:
            user_email: User identifier (used as agent_id in Mem0)
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            True if successful, False otherwise
        """
        if not self._ensure_client():
            return False
        
        if not messages or len(messages) == 0:
            logger.warning("⚠️ [MEMORY] No messages provided to store")
            return False
        
        try:
            # Convert messages to Mem0 format if needed
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    # Ensure proper format: {"role": "user"/"assistant", "content": "..."}
                    role = msg.get("role") or msg.get("user") or "user"
                    content = msg.get("content") or msg.get("assistant") or msg.get("message", "")
                    if content:
                        formatted_messages.append({
                            "role": role,
                            "content": content
                        })
            
            if not formatted_messages:
                logger.warning("⚠️ [MEMORY] No valid messages to store after formatting")
                return False
            
            # Add memories to Mem0
            # Mem0 will automatically extract and store relevant memories from the conversation
            self.client.add(
                messages=formatted_messages,
                user_id=user_email
            )
            
            logger.info(f"✅ [MEMORY] Stored {len(formatted_messages)} messages as memories for user {user_email}")
            return True
        except Exception as e:
            logger.error(f"❌ [MEMORY] Error storing memories for {user_email}: {e}", exc_info=True)
            return False
    
    async def get_user_profile(self, user_email: str) -> Dict[str, Any]:
        """
        Retrieve user profile/facts built from all interactions.
        
        Args:
            user_email: User identifier
            
        Returns:
            Dictionary containing user profile information
        """
        if not self._ensure_client():
            return {}
        
        try:
            # Get all memories for the user to build profile
            all_memories = self.client.get_all(user_id=user_email)
            
            # Extract profile information from memories
            profile = {
                "user_id": user_email,
                "total_memories": len(all_memories),
                "memories": all_memories
            }
            
            logger.info(f"✅ [MEMORY] Retrieved profile for user {user_email} with {len(all_memories)} memories")
            return profile
        except Exception as e:
            logger.error(f"❌ [MEMORY] Error retrieving profile for {user_email}: {e}", exc_info=True)
            return {}
    
    async def search_memories(
        self, 
        user_email: str, 
        query: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search memories by relevance (alias for get_memories for consistency).
        
        Args:
            user_email: User identifier
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of relevant memories
        """
        return await self.get_memories(user_email, query, limit)
    
    async def delete_user_memories(self, user_email: str) -> bool:
        """
        Delete all memories for a user (used in account deletion).
        
        Args:
            user_email: User identifier
            
        Returns:
            True if successful, False otherwise
        """
        if not self._ensure_client():
            return False
        
        try:
            db = get_db()
            collection_names = await db.list_collection_names()
            total_deleted = 0
            
            for collection_name in collection_names:
                # Target likely Mem0 collections
                if not (
                    collection_name in {"memories", "mem0_memories", "mem0_agents"}
                    or collection_name.startswith("mem0")
                    or collection_name.startswith("memory")
                ):
                    continue
                
                collection = db[collection_name]
                result = await collection.delete_many({
                    "$or": [
                        {"user_id": user_email},
                        {"agent_id": user_email},
                        {"metadata.user_id": user_email}
                    ]
                })
                deleted_count = result.deleted_count
                if deleted_count > 0:
                    total_deleted += deleted_count
                    logger.info(f"✅ [MEMORY] Deleted {deleted_count} memories from {collection_name} for user {user_email}")
            
            if total_deleted == 0:
                logger.warning(f"⚠️ [MEMORY] No memories found to delete for user {user_email}")
            else:
                logger.info(f"✅ [MEMORY] Deleted {total_deleted} total memories for user {user_email}")
            return True
        except Exception as e:
            logger.error(f"❌ [MEMORY] Error deleting memories for {user_email}: {e}", exc_info=True)
            return False


_memory_service_instance = None


def get_memory_service() -> MemoryService:
    """Get or create MemoryService instance (singleton)"""
    global _memory_service_instance
    
    if _memory_service_instance is None:
        _memory_service_instance = MemoryService()
    
    return _memory_service_instance

