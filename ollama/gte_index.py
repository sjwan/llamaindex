import chromadb

from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import Settings
from chromadb.utils import embedding_functions
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import chromadb.utils.embedding_functions as embedding_functions


ollama_embedding = OllamaEmbedding(
    model_name="gte",
    base_url="http://10.91.3.116:11434"
)


documents = SimpleDirectoryReader("./data1").load_data()

db = chromadb.PersistentClient(path="./chroma_db", )

# create collection
chroma_collection = db.get_or_create_collection("quickstart", 
            embedding_function= embedding_functions.OllamaEmbeddingFunction(
        model_name="gte",
    url="http://10.91.3.116:11434"
),
            metadata={"hnsw:space": "cosine"}
)

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
Settings.embed_model = ollama_embedding

# create your index
index = VectorStoreIndex.from_documents(
    documents, show_progress=True, storage_context=storage_context
)

index_retrive = index.as_retriever()

result = index_retrive.retrieve("投标邀请包含哪些内容？")

print (result)


