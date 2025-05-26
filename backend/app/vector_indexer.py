from typing import List, Optional, Dict, Any
import logging
import numpy as np
from redis import Redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redisvl.utils.vectorize.text.huggingface import HFTextVectorizer

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into chunks of approximately chunk_size characters."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

logger = logging.getLogger(__name__)

class DocumentVectorIndexer:
    """
    Simplified Redis vector indexer with semantic search capabilities.
    """
    
    def __init__(
        self,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        model_name: str = "sentence-transformers/all-mpnet-base-v2"
    ):
        self.redis = Redis(host=redis_host, port=redis_port, decode_responses=False)
        self.vectorizer = HFTextVectorizer(model=model_name)
        self.embedding_dim = 768  # Default for all-mpnet-base-v2
        
    def create_index(self, project_id: str):
        """Create a vector index for the project if it doesn't exist"""
        index_name = f"rag:{project_id}"
        prefix = f"{index_name}:"
        
        try:
            self.redis.ft(index_name).info()
            logger.info(f"Index {index_name} already exists")
            return
        except Exception:
            logger.info(f"Creating new index {index_name}")
            
        # Define schema
        schema = (
            TextField("id"),
            TextField("text"),
            VectorField(
                "embedding",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": self.embedding_dim,
                    "DISTANCE_METRIC": "COSINE"
                }
            )
        )
        
        # Create index
        definition = IndexDefinition(prefix=[prefix], index_type=IndexType.HASH)
        self.redis.ft(index_name).create_index(fields=schema, definition=definition)
        logger.info(f"Created index {index_name}")
    
    def add_document(self, project_id: str, text: str, doc_id: Optional[str] = None) -> str:
        """Add a document to the vector store"""
        if not doc_id:
            doc_id = f"doc:{len(self.redis.keys(f'rag:{project_id}:*')) + 1}"
            
        # Generate embedding
        embedding = self.vectorizer.embed(text)
        
        # Store in Redis
        key = f"rag:{project_id}:{doc_id}"
        self.redis.hset(
            key,
            mapping={
                "id": doc_id,
                "text": text,
                "embedding": np.array(embedding).astype(np.float32).tobytes()
            }
        )
        return key
    
    def search_similar_chunks(self, query: str, project_id: str, top_k: int = 3) -> List[str]:
        """Search for similar text chunks using semantic search"""
        index_name = f"rag:{project_id}"
        
        try:
            # Check if index exists
            self.redis.ft(index_name).info()
        except Exception as e:
            logger.warning(f"Index {index_name} does not exist: {e}")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.vectorizer.embed(query)
            
            # Prepare query
            query_vector = np.array(query_embedding).astype(np.float32).tobytes()
            query = "*=>[KNN {} @embedding $vector AS score]".format(top_k)
            
            # Execute query
            results = self.redis.ft(index_name).search(
                query,
                query_params={"vector": query_vector}
            )
            
            return [doc["text"] for doc in results.docs]
            
        except Exception as e:
            logger.error(f"Error searching index {index_name}: {e}")
            return []
            
    def check_index_health(self, project_id: str) -> Dict:
        """Check if the index exists and has documents"""
        try:
            index = self.redis.ft(f"rag:{project_id}")
            info = index.info()
            return {
                "exists": True,
                "num_docs": int(info.get('num_docs', 0)),
                "index_definition": info
            }
        except Exception as e:
            return {
                "exists": False,
                "error": str(e)
            }
