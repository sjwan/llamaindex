import chromadb

from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import Settings
from chromadb.utils import embedding_functions
from llama_index.llms.ollama import Ollama

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.postprocessor.jinaai_rerank import JinaRerank


jinaai_api_key = "jina_7e76397cb33648bf892b59c8e137453ckRPxkvRnIh4CxCMaRp-iQCuCW-7L"
model_name = "jina-embeddings-v2-base-zh"
embed_model = JinaEmbedding(
    api_key=jinaai_api_key,
    model=model_name,
)
jina_rerank = JinaRerank(api_key=jinaai_api_key, top_n=2)

db = chromadb.PersistentClient(path="./chroma_db", )

# create collection
chroma_collection = db.get_or_create_collection("quickstart", 
            embedding_function= embedding_functions.JinaEmbeddingFunction(
                api_key=jinaai_api_key,
                model_name=model_name
            ),
            metadata={"hnsw:space": "cosine"}
)

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
Settings.embed_model = embed_model

# create your index
index = VectorStoreIndex.from_vector_store(
    vector_store, show_progress=True, storage_context=storage_context
)

llm = Ollama(model="qwen2:7b", request_timeout=120.0)
Settings.llm = llm


query_engine = index.as_query_engine(similarity_top_k=10, node_postprocessors=[jina_rerank])

response = query_engine.query("项目编号是什么？")

print(response)
