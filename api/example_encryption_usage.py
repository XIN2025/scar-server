"""
Example usage of the EncryptionService with MongoDB documents and Pydantic schemas.
"""

from app.services.backend_services.encryption_service import EncryptionService, get_encryption_service
from app.schemas.ai.goals import GoalCreate, Goal
from datetime import datetime

def example_usage():
    """Example of how to use the encryption service"""
    
    # Initialize encryption service
    encryption_service = EncryptionService()
    
    # Sample MongoDB document (as it would come from/go to database)
    goal_document = {
        "user_email": "user@example.com",
        "title": "Learn Python Programming",  # This will be encrypted
        "description": "Complete Python course by end of month",
        "priority": "high",
        "category": "personal",
        "id": "goal_123",
        "created_at": datetime.now()
    }
    
    print("Original document:")
    print(goal_document)
    print()
    
    # Get encryption info for the schema
    encryption_info = encryption_service.get_encryption_info(GoalCreate)
    print("Encryption info:")
    print(encryption_info)
    print()
    
    # Encrypt the document before storing in MongoDB
    encrypted_document = encryption_service.encrypt_document(goal_document, GoalCreate)
    print("Encrypted document:")
    print(encrypted_document)
    print()
    
    # Decrypt the document after retrieving from MongoDB
    decrypted_document = encryption_service.decrypt_document(encrypted_document, GoalCreate)
    print("Decrypted document:")
    print(decrypted_document)
    print()
    
    # Verify encryption/decryption worked
    assert decrypted_document["title"] == goal_document["title"]
    print("✅ Encryption/Decryption successful!")
    
    # Example with bulk operations
    documents = [goal_document, goal_document.copy()]
    documents[1]["title"] = "Learn Machine Learning"
    
    encrypted_docs = encryption_service.encrypt_documents_bulk(documents, GoalCreate)
    decrypted_docs = encryption_service.decrypt_documents_bulk(encrypted_docs, GoalCreate)
    
    print(f"✅ Bulk operations successful! Processed {len(documents)} documents")

if __name__ == "__main__":
    example_usage()