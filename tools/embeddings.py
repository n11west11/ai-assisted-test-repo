import os

from dotenv import load_dotenv
from langchain.document_loaders.mongodb import MongodbLoader
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.embeddings import OpenAIEmbeddings
from pymongo import MongoClient

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MONGO_URI = os.environ["MONGO_URI"]
DB_NAME = os.environ["MONGO_DATABASE_NAME"]
COLLECTION_NAME = os.environ["COLLECTION_NAME"]
# client = MongoClient(MONGO_URI)
# db = client[DB_NAME]
# MONGODB_COLLECTION = db[COLLECTION_NAME]

# loader = MongodbLoader(
#     connection_string=MONGO_URI,
#     db_name=DB_NAME,
#     collection_name=COLLECTION_NAME
# )

# docs = loader.load()

underlying_embeddings = OpenAIEmbeddings()

store = LocalFileStore("./cache/")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=underlying_embeddings.model
)