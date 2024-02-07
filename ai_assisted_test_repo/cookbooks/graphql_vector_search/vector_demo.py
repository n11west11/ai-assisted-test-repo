import os

from dotenv import load_dotenv
from langchain.document_loaders.mongodb import MongodbLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from pymongo import MongoClient

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MONGO_URI = os.environ["MONGO_URI"]
DB_NAME = os.environ["MONGO_DATABASE_NAME"]
COLLECTION_NAME = os.environ["COLLECTION_NAME"]
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
MONGODB_COLLECTION = db[COLLECTION_NAME]

loader = MongodbLoader(
    connection_string=MONGO_URI,
    db_name=DB_NAME,
    collection_name=COLLECTION_NAME
)

docs = loader.load()

len(docs)

try: 
    db = FAISS.load_local("faiss_index", embeddings=OpenAIEmbeddings(disallowed_special=()))
except:
    db = FAISS.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(disallowed_special=())
    )

db.save_local("faiss_index")

query = "What queries can find ships?"

results = db.similarity_search(
    query=query, k=5
)

# Display results
for result in results:
    print(result.page_content + "\n")
