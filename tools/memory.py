import os
import time
from datetime import datetime
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions
from typing import Dict, List
import logging
import config
from typing import Dict, List

MEMORY_STORE_NAME = "agent-memory"

#default_ef = embedding_functions.DefaultEmbeddingFunction()

ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name=config.MODEL,
)
ollama_ef._session.timeout = 1200  


# Memory storage using local ChromaDB
class DefaultMemoryStorage:
    def __init__(self):
        logging.getLogger('chromadb').setLevel(logging.ERROR)
        # Create Chroma collection
        chroma_persist_dir = ".chroma"
        chroma_client = chromadb.PersistentClient(path=chroma_persist_dir)        

        metric = "cosine"        
        self.collection = chroma_client.get_or_create_collection(
            name=MEMORY_STORE_NAME,
            metadata={"hnsw:space": metric},            
            #embedding_function=default_ef,
            embedding_function=ollama_ef,
        )
        count = self.collection.count()
        print (f"{count} Records in long-term memory (.chroma)")

    def add(self, agent_name, content, threshold=0.1):
        # Check for duplicates using similarity search
        timestr = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        if agent_name:
            results = self.collection.query(
                query_texts=[content],
                n_results=1,
                where={"agent_name": agent_name}
            )
            ids=[f"{agent_name}-{timestr}"]
        else:
            results = self.collection.query(
                query_texts=[content],
                n_results=1,             
            )
            ids=[f"{timestr}"]
            agent_name=""
                        
        # check for edge case when db is empty
        #count = self.collection.count()
        if len(results['distances'][0]) > 0:
            distance = results['distances'][0][0]
        else:
            distance = threshold + 1.0

        if distance > threshold:            
            self.collection.add(
                documents=[content],                
                metadatas=[{"agent_name": agent_name, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}],
                ids=ids
            )
        else:
            #print("Duplicate content detected. Not adding to storage.")
            pass            

    def recall(self, query, n_results=10, agent_name=None, threshold=0.1):
        if agent_name:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"agent_name": agent_name}
            )
        else:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )            

        # Filter results based on minimum distance threshold
        filtered_results = [
            (doc, dist, metadata) for doc, dist, metadata in zip(results['documents'][0], results['distances'][0], results['metadatas'][0])
            if dist < threshold
        ]

        # Sort by distance if needed (optional)
        filtered_results.sort(key=lambda x: x[1], reverse=False)

        # Return list of dictionaries
        return [{'document': doc, 'distance': "{:.3f}".format(dist), 'agent_name': metadata["agent_name"], 'timestamp': metadata["timestamp"]} for doc, dist, metadata in filtered_results]

memory_storage = None

# Initialize memory storage
def init_long_term_memory():
    global memory_storage
    memory_storage = DefaultMemoryStorage()

# internal functions
def _store_memory(agent_name: str, text: str, threshold=0.1):
    if memory_storage is None:
        print("Error: please call init_long_term_memory() to init the memory.")
    memory_storage.add(agent_name, text, threshold)
    
def _recall_memory(query: str, top_results_num: int, agent_name: str = None, threshold=0.9) -> List[Dict[str, str]]:
    if memory_storage is None:
        print("Error: please call init_long_term_memory() to init the memory.")
    return memory_storage.recall(query, top_results_num, agent_name, threshold) 

# public tools
def store_in_memory_tool(agent_name: str, text: str) -> str:
    ''' Store information in long-term memory '''
    print("Using Tool: store_in_memory")
    _store_memory(agent_name, text=text)
    return "Memory was stored successfully."

def recall_from_memory_tool(query: str) -> str:
    ''' Recall information from long-term memory '''
    print("Using Tool: recall_from_memory")
    docs = _recall_memory(query, 3, None, 0.9)
    if len(docs) > 0:
        return str(docs)
    return "Sorry, nothing found."
