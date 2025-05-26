"""
Main application file for the Binod AI Assistant backend.

This module sets up the FastAPI application, including routes for chat,
file ingestion, and document upload. It also configures CORS and initializes
the necessary components for the application to run.
"""

from fastapi import FastAPI, WebSocket, Request, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.websocket_chat import chat_endpoint
from app.vector_indexer import vector_indexer
import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from storage3.exceptions import StorageApiError
from typing import List
from app.shared_resources import *

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()


# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],              # allow all origins for development
    allow_credentials=False,           # no cookies/auth needed here
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for new chat"""
    await chat_endpoint(websocket)

@app.websocket("/chat/{thread_id}")
async def websocket_endpoint_with_thread(websocket: WebSocket, thread_id: str):
    """WebSocket endpoint for chat with thread ID"""
    await chat_endpoint(websocket, thread_id)


@app.post("/ingest")
async def ingest_documents(request: Request):
    """Ingest text into Redis vector index for a project."""
    data = await request.json()
    project_id = data.get("project_id")
    text = data.get("text")
    if not project_id or not text:
        raise HTTPException(status_code=400, detail="project_id and text are required")
    logger.info(f"Received ingestion request: project_id={project_id}, text_length={len(text)}")
    count = vector_indexer.ingest(project_id, text)
    logger.info(f"Indexed {count} chunks for project_id={project_id}")
    return {"ingested_chunks": count}

@app.post("/upload-document")
async def upload_document(conversation_id: str = Form(...), file: UploadFile = File(...)):
    """Upload a document, extract text, index into Redis, and store in Supabase."""
    # Save to temp file
    suffix = Path(file.filename).suffix
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    logger.info(f"Saved upload to {tmp_path}")
    # Load document
    if file.content_type == "application/pdf":
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
    elif file.content_type in ["text/plain", "text/markdown"]:
        loader = TextLoader(tmp_path, encoding="utf-8")
        documents = loader.load()
    elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        loader = Docx2txtLoader(tmp_path)
        documents = loader.load()
    else:
        raise HTTPException(status_code=415, detail="Unsupported file type")
    # Chunk text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)
    logger.info(f"Split document into {len(docs)} chunks")
    # Index text chunks via shared vector_indexer
    text_chunks = [doc.page_content for doc in docs]
    count = vector_indexer.ingest_chunks(conversation_id, text_chunks)
    logger.info(f"Indexed {count} chunks for conversation {conversation_id}")
    # Cleanup temp file
    os.remove(tmp_path)
    # Upload to Supabase storage (ensure bucket exists)
    bucket_id = "chat-files"
    storage = supabase.storage
    path = f"{conversation_id}/{file.filename}"
    try:
        bucket = storage.from_(bucket_id)
        bucket.upload(path, contents)
    except StorageApiError as e:
        # Create bucket if missing
        if getattr(e, 'statusCode', None) == 404 or "Bucket not found" in str(e):
            storage.create_bucket(bucket_id, public=True)
            bucket = storage.from_(bucket_id)
            bucket.upload(path, contents)
        else:
            raise HTTPException(status_code=500, detail=f"Storage upload error: {e}")
    # get_public_url returns a direct URL string
    public_url = bucket.get_public_url(path)
    logger.info(f"Uploaded file to Supabase: {public_url}")
    # Store metadata in Supabase
    try:
        supabase.table("documents").insert({
            "conversation_id": conversation_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "chunk_count": count,
            "size_bytes": len(contents)
        }).execute()
    except Exception as e:
        logger.error(f"Failed to store document metadata: {e}")
        # Don't fail the request if metadata storage fails
    return {"status": "success", "chunks_indexed": count, "conversation_id": conversation_id}
