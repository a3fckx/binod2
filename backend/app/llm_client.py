"""
Simplified LLM Client for OpenRouter with Redis Semantic Caching
"""
import os
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain_community.cache import RedisSemanticCache
from langchain_huggingface import HuggingFaceEmbeddings


# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ChatOpenRouter(ChatOpenAI):
    """Simplified OpenRouter chat model with OpenAI compatibility"""
    
    def __init__(self, **kwargs):
        # First initialize the parent class
        super().__init__(
            model=os.getenv("OPENROUTER_MODEL", "microsoft/mai-ds-r1:free"),
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": os.getenv("APP_URL", "http://localhost:3000"),
                "X-Title": os.getenv("APP_NAME", "Binod AI Assistant")
            },
            **kwargs
        )
        
        # Initialize Redis Semantic Cache after parent is initialized
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        
        try:
            # Initialize embeddings with a lightweight model
            model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU is available
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Create cache with required parameters
            redis_cache = RedisSemanticCache(
                redis_url=redis_url,
                embedding=embeddings
            )
            
            set_llm_cache(redis_cache)
            # Get the model name safely
            model_name = getattr(self, 'model', 'unknown')
            logger.info(f"Initialized Redis semantic cache for model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Redis semantic cache: {e}")
            logger.warning("Proceeding without cache")

# Create and export LLM instance
llm = ChatOpenRouter()
