import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI","mongodb+srv://rajesh:rajesh@cluster0.ejwfzgv.mongodb.net/")
DB_NAME = os.getenv("MONGODB_DB_NAME", "scar")
VECTOR_STORE_DB_URI = os.getenv("VECTOR_STORE_DB_URI", "test_vector_store_uri")
VECTOR_COLLECTION_NAME=os.getenv("VECTOR_COLLECTION_NAME", "test_vector_collection")
VECTOR_DB_NAME=os.getenv("VECTOR_DB_NAME", "test_vector_db")
SECRET_KEY = os.getenv("SECRET_KEY", "secret")
NUDGE_SCHEDULED_COLLECTION = os.getenv("NUDGE_SCHEDULED_COLLECTION", "nudge_scheduled_jobs")

# SMTP/Email Configuration
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_FROM = os.getenv("SMTP_FROM")
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "JetFuel")

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "1234qazw0987")

# Vector Store Configuration
VECTOR_INDEX_PATH = os.getenv("VECTOR_INDEX_PATH", "faiss_index.bin")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# LLM Configuration
OPENAI_API_KEY = os.getenv("OPENAI_KEY", "test-openai-key")  # Load from OPENAI_KEY in .env
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.1")  
LLM_TEMPERATURE = os.getenv("LLM_TEMPERATURE", "0.2")

PERSONALIZATION_MODEL = os.getenv("PERSONALIZATION_MODEL", "ft:o4-mini-2025-04-16:evra-health:user-goals:CfOWayPc")

 

# File Upload Configuration
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx", ".doc"}
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
PROFILE_PICTURE_MAX_BYTES = int(os.getenv("PROFILE_PICTURE_MAX_BYTES", "10485760"))
ALLOWED_PROFILE_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

MULTIPART_SPOOL_MAX_SIZE = int(os.getenv("MULTIPART_SPOOL_MAX_SIZE", "1048576"))   
MULTIPART_MAX_FILES = int(os.getenv("MULTIPART_MAX_FILES", "10"))  
MULTIPART_MAX_FIELDS = int(os.getenv("MULTIPART_MAX_FIELDS", "20"))  

DEBUG_COLLECTION_NAME = os.getenv("DEBUG_COLLECTION_NAME", "debug_data")

# Firebase Configuration
# Default to checking in api directory root if env var not set
# Get the api directory (parent of app directory where config.py is located)
_api_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FIREBASE_SERVICE_ACCOUNT_PATH = os.path.join(_api_dir, "")
# Get env var if set, but don't use it as default (will be validated in nudge_service)
FIREBASE_SERVICE_ACCOUNT_PATH_ENV = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")

# Account Deletion Configuration
APP_BASE_URL = os.getenv("APP_BASE_URL", "https://your-app.com")  # Base URL for email links

# Mem0 Configuration
MEM0_API_KEY = os.getenv("MEM0_API_KEY")  # Optional: for hosted Mem0 platform
MEM0_STORAGE_TYPE = os.getenv("MEM0_STORAGE_TYPE", "mongodb")  # mongodb or postgres 
