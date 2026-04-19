from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from app.config import MONGO_URI, MONGO_DB

client: AsyncIOMotorClient = None
db = None
gridfs_bucket: AsyncIOMotorGridFSBucket = None


async def connect_db():
    global client, db, gridfs_bucket
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[MONGO_DB]
    gridfs_bucket = AsyncIOMotorGridFSBucket(db, bucket_name="images")


async def close_db():
    if client:
        client.close()


def get_db():
    return db


def get_gridfs():
    return gridfs_bucket
