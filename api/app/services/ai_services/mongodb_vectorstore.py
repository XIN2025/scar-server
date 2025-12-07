from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from pymongo import MongoClient
import logging
import json
from typing import Dict, List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4

from ...config import VECTOR_STORE_DB_URI, VECTOR_DB_NAME, VECTOR_COLLECTION_NAME, OPENAI_API_KEY, EMBEDDING_MODEL
from app.schemas.backend.documents import DocumentType

logger = logging.getLogger(__name__)

class MongoVectorStoreService:
    def __init__(self):
        self.client = MongoClient(VECTOR_STORE_DB_URI)
        self.db = self.client[VECTOR_DB_NAME]
        self.collection = self.db[VECTOR_COLLECTION_NAME]

        logger.info(f"Initializing OpenAIEmbeddings with model: {EMBEDDING_MODEL}")
        self.embedding_fn = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )

        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embedding_fn,
            index_name="vector_index",
            text_key="text",
            embedding_key="embedding",
        )
        logger.info("MongoVectorStoreService initialized successfully with OpenAI embeddings.")

    def add_document(self, content: str, user_email: str, filename: str, type: DocumentType, upload_id: str = None, chunk_size: int = 1500, chunk_overlap: int = 150) -> int:
        if upload_id is None:
            upload_id = str(uuid4())
        
        doc = Document(
            page_content=content,
            metadata={
                "user_email": user_email,
                "filename": filename,
                "size": len(content),
                "upload_id": upload_id,
                "type": type.value 
            },
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents([doc])
        logger.info(f"Document '{filename}' split into {len(chunks)} chunks.")

        if not chunks:
            logger.warning(f"Document '{filename}' produced no chunks to add.")
            return 0
        
        try:
            # OpenAIEmbeddings can handle batching internally, but for clarity and control,
            # we'll still do it this way.
            texts_to_embed = [chunk.page_content for chunk in chunks]
            embeddings = self.embedding_fn.embed_documents(texts_to_embed)
            
            records_to_insert = []
            for i, chunk in enumerate(chunks):
                record = {
                    "text": chunk.page_content,
                    "embedding": embeddings[i],
                    **chunk.metadata
                }
                records_to_insert.append(record)
            
            self.collection.insert_many(records_to_insert)
            logger.info(f"Successfully inserted {len(records_to_insert)} chunks for '{filename}' into collection '{VECTOR_COLLECTION_NAME}'.")
        except Exception as e:
            logger.error(f"❌ Failed to insert document chunks for '{filename}': {e}", exc_info=True)
            raise

        return len(chunks)
    
    def get_all_documents_by_user_email(self, user_email: str) -> List[Dict]:
        try:
            cursor = self.collection.find(
                 {"user_email": user_email, "upload_id": {"$exists": True, "$ne": None}}, 
                {"embedding": 0, "_id": 0}
            )
            unique_docs = {}
            for doc in cursor:
                upload_id = doc.get('upload_id')
                if upload_id:
                    if upload_id not in unique_docs:
                        unique_docs[upload_id] = doc
            return list(unique_docs.values())
        except Exception as e:
            logger.error(f"❌ Failed fetching document list for {user_email}: {e}")
            return []

    def get_document_info_by_upload_id(self, upload_id: str) -> Optional[Dict]:
        """Get document metadata (filename, user_email) by upload_id"""
        try:
            doc = self.collection.find_one(
                {"upload_id": upload_id},
                {"embedding": 0, "text": 0, "_id": 0}  # Exclude large fields
            )
            if doc:
                return {
                    "filename": doc.get("filename"),
                    "user_email": doc.get("user_email"),
                    "type": doc.get("type")
                }
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get document info for upload_id '{upload_id}': {e}")
            return None

    def delete_document_by_upload_id(self, upload_id: str) -> bool:
        try:
            result = self.collection.delete_many({"upload_id": upload_id})
            logger.info(f"Deleted {result.deleted_count} chunks for upload_id '{upload_id}'.")
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"❌ Failed deleting document chunks for upload_id '{upload_id}': {e}")
            return False

    def search(self, query: str, user_email: str, top_k: int = 5) -> List[Dict]:
        logger.info(f"🔍 [VECTOR SEARCH] Starting vector search for query: '{query}' (User: {user_email})")
        
        if not query or not query.strip():
            logger.warning("🔍 [VECTOR SEARCH] Query is empty or None, returning empty list")
            return []

        try:
            # Check if documents exist for this user
            doc_count = self.collection.count_documents({"user_email": user_email})
            logger.info(f"🔍 [VECTOR SEARCH] Found {doc_count} total documents for user {user_email} in collection")
            
            if doc_count == 0:
                logger.warning(f"⚠️ [VECTOR SEARCH] No documents found for user {user_email}")
                return []
            
            # Check lab report count specifically
            lab_report_count = self.collection.count_documents({"user_email": user_email, "type": "lab_report"})
            logger.info(f"🔍 [VECTOR SEARCH] Found {lab_report_count} lab report chunks for user {user_email}")
            
            search_filter = {"user_email": {"$eq": user_email}}
            logger.info(f"🔍 [VECTOR SEARCH] Stage 1: Constructed search filter: {json.dumps(search_filter)}")

            logger.info(f"🔍 [VECTOR SEARCH] Stage 2: Calling LangChain's similarity_search with k={top_k}...")
            
            docs = self.vector_store.similarity_search(
                query,
                k=top_k,
                pre_filter=search_filter
            )

            logger.info(f"🔍 [VECTOR SEARCH] Stage 3: LangChain call complete. Retrieved {len(docs)} document(s).")
            
            results = []
            if docs:
                for i, d in enumerate(docs):
                    text = getattr(d, "page_content", "")
                    metadata = getattr(d, "metadata", {})
                    if not text.strip(): 
                        logger.warning(f"⚠️ [VECTOR SEARCH] Skipping document {i} - empty text content")
                        continue
                    result = {"text": text, **metadata}
                    results.append(result)
                    logger.info(f"✅ [VECTOR SEARCH] Doc {i}: type={metadata.get('type')}, filename={metadata.get('filename')}, text_len={len(text)}")

            logger.info(f"✅ [VECTOR SEARCH] Completed successfully, returning {len(results)} valid results.")
            return results

        except Exception as e:
            logger.error(f"❌ [VECTOR SEARCH] A critical error occurred during the search operation: {e}", exc_info=True)
            # Try a fallback search without vector similarity if the index isn't working
            try:
                logger.info(f"🔄 [VECTOR SEARCH] Attempting fallback: direct MongoDB text search")
                # Fallback: just retrieve recent documents for this user
                fallback_docs = list(self.collection.find(
                    {"user_email": user_email},
                    {"text": 1, "user_email": 1, "filename": 1, "type": 1, "upload_id": 1}
                ).limit(top_k))
                
                logger.info(f"🔄 [VECTOR SEARCH] Fallback retrieved {len(fallback_docs)} documents")
                return [{"text": doc.get("text", ""), **{k: v for k, v in doc.items() if k != "_id" and k != "text"}} for doc in fallback_docs]
            except Exception as fallback_error:
                logger.error(f"❌ [VECTOR SEARCH] Fallback search also failed: {fallback_error}")
                return []

    def get_stats(self):
        return {"total_nodes": self.collection.count_documents({})}

vector_store = None

def get_vector_store() -> MongoVectorStoreService:
    global vector_store
    if vector_store is None:
        vector_store = MongoVectorStoreService()
    return vector_store
