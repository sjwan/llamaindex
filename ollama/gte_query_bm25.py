import chromadb

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import Settings
from chromadb.utils import embedding_functions
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import chromadb.utils.embedding_functions as embedding_functions

from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.query_engine import RetrieverQueryEngine



from llama_index.core.tools import RetrieverTool
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


from llama_index.core.retrievers import BaseRetriever


class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes

'''
“结巴”中文分词 
'''
import jieba
from typing import List

def chinese_tokenizer(text_list: list) -> List[str]:
    # Use jieba to segment Chinese text
    tokens = [list(jieba.cut(text)) for text in text_list]
    return [item for sublist in tokens for item in sublist]

'''
切分文档，并创建Nodes
'''
documents = SimpleDirectoryReader(input_files=["/Users/hawk/workspace/projects/llm/index100/llamaindex/ollama/data1/a2.md"]).load_data()
splitter = SentenceSplitter(chunk_size=512)
nodes = splitter.get_nodes_from_documents(documents)


ollama_embedding = OllamaEmbedding(
    model_name="gte",
    base_url="http://10.91.3.116:11434"
)
llm = Ollama(
    base_url="http://10.91.3.116:11434", 
    model="qwen2:7b", 
    request_timeout=120.0)
Settings.llm = llm
Settings.embed_model = ollama_embedding



db = chromadb.PersistentClient(path="./chroma_db", )
chroma_collection = db.get_or_create_collection("quickstart", 
            embedding_function= embedding_functions.OllamaEmbeddingFunction(
                model_name="gte",
                url="http://10.91.3.116:11434"
            ),
            metadata={"hnsw:space": "cosine"}
)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# create your index
index = VectorStoreIndex.from_vector_store(
    vector_store, show_progress=True, storage_context=storage_context
)


vector_retriever = VectorIndexRetriever(index, similarity_top_k=2)

bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2, stemmer=chinese_tokenizer, verbose=True)

hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)



query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    llm=llm,
)


# query_engine = index.as_query_engine(similarity_top_k=10,)

response = query_engine.query("项目编号多少？")

print(response)
