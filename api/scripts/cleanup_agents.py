from pymongo import MongoClient
from app.config import MONGODB_URI

def cleanup_agents():
    print("Connecting to MongoDB...")
    client = MongoClient(MONGODB_URI)
    db = client.get_default_database()
    
    print("Dropping agents collection...")
    if "agents" in db.list_collection_names():
        db.agents.drop()
        print("Agents collection dropped successfully")
    else:
        print("Agents collection not found")
    
    client.close()
    print("Cleanup complete")

if __name__ == "__main__":
    cleanup_agents()
