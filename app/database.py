"""
MongoDB database connection and initialization
"""
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from app.config import get_settings

settings = get_settings()


class Database:
    client: AsyncIOMotorClient = None
    db: AsyncIOMotorDatabase = None
    collection: AsyncIOMotorCollection = None


db = Database()


async def connect_to_mongo():
    """Connect to MongoDB"""
    db.client = AsyncIOMotorClient(settings.MONGO_URI)
    db.db = db.client[settings.DB_NAME]
    db.collection = db.db[settings.COLLECTION_NAME]
    
    # Create indexes
    await create_indexes()
    print(f"Connected to MongoDB: {settings.DB_NAME}")


async def close_mongo_connection():
    """Close MongoDB connection"""
    if db.client:
        db.client.close()
        print("Closed MongoDB connection")


async def create_indexes():
    """Create necessary indexes"""
    await db.collection.create_index([("unique_id", 1)], unique=True)
    await db.collection.create_index([("metadata.category", 1)])
    await db.collection.create_index([("metadata.source_type", 1)])
    await db.collection.create_index([("created_at", -1)])
    print("Indexes created successfully")


def get_collection() -> AsyncIOMotorCollection:
    """Get the MongoDB collection"""
    return db.collection