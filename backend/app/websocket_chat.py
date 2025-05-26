from fastapi import WebSocket, WebSocketDisconnect
from redisvl.extensions.session_manager import SemanticSessionManager
import json
from datetime import datetime
import asyncio
import logging
from typing import Dict, Any, List, Optional
from app.shared_resources import redis_client
from app.agent_system import create_conversation_thread, process_message

logger = logging.getLogger(__name__)


class ChatManager:
    def __init__(self):
        logger.info("ChatManager initialized")
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_manager = SemanticSessionManager(
            name='binod_chat',
            redis_client=redis_client
        )
        self.thinking_steps_cache: Dict[str, List[str]] = {}
        
    async def connect(self, websocket: WebSocket, thread_id: str = None):
        logger.info("Accepting WebSocket connection")
        await websocket.accept()
        if not thread_id:
            thread_id = create_conversation_thread()
        logger.info(f"Thread {thread_id}: Connection established")
        self.active_connections[thread_id] = websocket
        return thread_id
        
    def disconnect(self, thread_id: str):
        if thread_id in self.active_connections:
            del self.active_connections[thread_id]

    async def send_thinking_step(self, websocket: WebSocket, step: str, thread_id: Optional[str] = None):
        """Send a thinking step to the client"""
        message = {
            "type": "thinking_step",
            "content": step
        }
        await websocket.send_json(message)
        
        # Cache thinking step if thread_id is provided
        if thread_id:
            if thread_id not in self.thinking_steps_cache:
                self.thinking_steps_cache[thread_id] = []
            self.thinking_steps_cache[thread_id].append(step)
            
        logger.info(f"Sent thinking step: {step}")

    async def send_response(self, websocket: WebSocket, content: str, thinking_steps: List[str], thread_id: Optional[str] = None):
        """Send the final response to the client"""
        # If thread_id is provided and we have cached thinking steps, merge them with the provided ones
        if thread_id and thread_id in self.thinking_steps_cache:
            # Combine cached steps with new ones, avoiding duplicates
            all_steps = self.thinking_steps_cache[thread_id].copy()
            for step in thinking_steps:
                if step not in all_steps:
                    all_steps.append(step)
            thinking_steps = all_steps
            
            # Clear the cache after sending
            self.thinking_steps_cache.pop(thread_id, None)
        
        message = {
            "type": "response",
            "content": content,
            "thinking_steps": thinking_steps
        }
        await websocket.send_json(message)
        logger.info(f"Sent response: {content[:50]}...")

    async def get_chat_history(self, thread_id: str):
        try:
            # Get recent messages for the thread
            messages = self.session_manager.get_recent(top_k=10)
            return messages
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return []

    async def store_message(self, thread_id: str, role: str, content: str):
        try:
            self.session_manager.add_message({
                "role": role,
                "content": content,
                "thread_id": thread_id,
                "timestamp": datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.error(f"Error storing message: {e}")

chat_manager = ChatManager()

async def chat_endpoint(websocket: WebSocket, thread_id: str = None):
    thread_id = await chat_manager.connect(websocket, thread_id)
    logger.info(f"WebSocket opened for thread {thread_id}")
    
    try:
        # Send chat history if thread exists
        history = await chat_manager.get_chat_history(thread_id)
        logger.info(f"Thread {thread_id}: Retrieved history count = {len(history)}")
        if history:
            await websocket.send_json({
                "type": "history",
                "messages": history
            })
        
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            logger.info(f"Thread {thread_id}: Received message: {data.get('content','')}")
            content = data.get("content", "").strip()
            file_url = data.get("fileUrl", "")
            quote = data.get("quote", "")
            
            # Debug log the entire message data
            logger.info(f"Thread {thread_id}: Full message data: {data}")
            
            if not content:
                continue
                
            # Log if quote is present
            if quote:
                logger.info(f"Thread {thread_id}: Received quote: {quote[:50]}...")
            
            # Store user message
            await chat_manager.store_message(thread_id, "user", content)
            logger.info(f"Thread {thread_id}: Stored user message: {content}")
            
            # Initial thinking steps
            initial_steps = [
                "🧭 Analyzing your message...",
                "🔍 Determining how to best respond..."
            ]
            
            # Show initial thinking steps
            for step in initial_steps:
                await chat_manager.send_thinking_step(websocket, step, thread_id)
                logger.info(f"Thread {thread_id}: Sent thinking step: {step}")
                await asyncio.sleep(0.3)  # Short delay for UX
            
            try:
                # Process message through LangGraph agent
                response, updated_thinking_steps = await process_message(thread_id, content, quote)
                
                # Update thinking steps if provided
                if updated_thinking_steps and len(updated_thinking_steps) > 0:
                    thinking_steps = updated_thinking_steps
                    logger.info(f"Thread {thread_id}: Updated thinking steps from agent")
                
                # Store assistant response
                await chat_manager.store_message(thread_id, "assistant", response)
                logger.info(f"Thread {thread_id}: Stored assistant response")
                
                # Send final response with thinking steps
                await chat_manager.send_response(websocket, response, thinking_steps, thread_id)
                
            except Exception as e:
                error_msg = f"Error processing message: {str(e)}"
                logger.error(f"Thread {thread_id}: {error_msg}")
                await websocket.send_json({
                    "type": "error",
                    "content": error_msg
                })
            logger.info(f"Thread {thread_id}: Sent final response")
            
    except WebSocketDisconnect:
        chat_manager.disconnect(thread_id)
        logger.info(f"Thread {thread_id}: WebSocket disconnected")
    except Exception as e:
        logger.error(f"Thread {thread_id}: Error in chat endpoint: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "content": str(e)
            })
        except:
            logger.error(f"Thread {thread_id}: Could not send error message to client")
