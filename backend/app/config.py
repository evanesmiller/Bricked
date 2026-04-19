import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "bricked")
MAX_IMAGE_SIZE_MB = 50
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}
HEIC_EXTENSIONS = {".heic", ".heif"}
HEIC_MIME_TYPES = {"image/heic", "image/heif"}
