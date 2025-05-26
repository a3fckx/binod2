"""
Shared resources for the Binod AI Assistant backend.

This module provides centralized access to shared resources like database connections,
ensuring they're initialized only once and consistently used throughout the application.
"""

import os
import logging
from redis import Redis
from dotenv import load_dotenv
from supabase import create_client, Client
from app.vector_indexer import DocumentVectorIndexer
from app.llm_client import ChatOpenRouter

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Redis client
redis_client = Redis(
    host=os.getenv('REDIS_HOST', 'redis'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    decode_responses=True
)

logger.info("Shared Redis client initialized")

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

logger.info("Shared Supabase client initialized")

# Global instances
vector_indexer = DocumentVectorIndexer()
llm = ChatOpenRouter()

# For backward compatibility
llm_client = llm
