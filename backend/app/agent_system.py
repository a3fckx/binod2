"""
Enhanced Agent System with LangGraph, RAG, and Multi-level Memory
"""
from typing import List, Dict, Any, Optional, TypedDict, Annotated, Literal, Union, Callable
import logging
from datetime import datetime
import uuid
import json
import re
import math
from pydantic import BaseModel, Field

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage, BaseMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

# Local imports
from app.shared_resources import *

# Configure logging
logger = logging.getLogger(__name__)



# System prompts
SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's questions to the best of your ability.
Use the provided context when relevant, but you can also use your general knowledge.
If you don't know the answer, just say you don't know. Be concise and to the point.

Current conversation:
{history}

Context:
{context}

User's question: {question}"""

ROUTER_PROMPT = """You are a helpful AI assistant. Your task is to analyze the user's message and determine the best way to respond.
Choose one of the following options:
1. "rag" - If the user is asking a question that might benefit from retrieving specific information from our knowledge base
2. "calculator" - If the user is asking a math question or needs a calculation
3. "direct" - If the user is making a general request or asking a question that can be answered with general knowledge

User's message: {question}
Recent conversation history: {history}

Respond with just one word: "rag", "calculator", or "direct".
"""

CALCULATOR_PROMPT = """You are a calculator assistant. Solve the mathematical problem step by step.
Show your work clearly, explaining each step of the calculation.

Problem: {question}
"""

SUMMARIZER_PROMPT = """Summarize the following conversation history in a concise way that captures the main points.
Focus on key information that would be relevant for continuing the conversation.
Keep the summary under 200 words.

Conversation history:
{history}
"""

# Define state types
class AgentState(TypedDict):
    """State for our agent workflow"""
    messages: List[Dict[str, str]]
    context: str
    thinking_steps: List[str]
    history: str
    thread_id: str
    short_term_memory: List[Dict[str, str]]
    long_term_memory: List[Dict[str, str]]
    working_memory: Dict[str, Any]
    route: str
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    summary: str

# Helper functions
def log_step(state: AgentState, step: str) -> AgentState:
    """Helper function to log thinking steps"""
    step_with_timestamp = f"{datetime.now().strftime('%H:%M:%S')} - {step}"
    state["thinking_steps"].append(step_with_timestamp)
    logger.info(step)
    return state

def get_conversation_history(thread_id: str) -> List[Dict[str, str]]:
    """Get conversation history for a thread from Redis"""
    try:
        history = redis_client.get(f"conversation:{thread_id}")
        return json.loads(history) if history else []
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        return []

def update_conversation_history(thread_id: str, role: str, content: str):
    """Update conversation history in Redis"""
    try:
        history = get_conversation_history(thread_id)
        
        # Add new message
        history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only the last 20 messages
        history = history[-20:]
        
        # Save back to Redis
        redis_client.set(f"conversation:{thread_id}", json.dumps(history))
        
    except Exception as e:
        logger.error(f"Error updating conversation history: {e}")

def format_history(messages: List[Dict[str, str]]) -> str:
    """Format conversation history for prompt context"""
    if not messages:
        return "No conversation history."
    
    formatted = []
    for msg in messages:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        formatted.append(f"{role}: {content}")
    
    return "\n\n".join(formatted)

def messages_to_langchain_messages(messages: List[Dict[str, str]]) -> List[BaseMessage]:
    """Convert message dicts to LangChain message objects"""
    lc_messages = []
    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        elif role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "function":
            lc_messages.append(FunctionMessage(content=content, name=msg.get("name", "function")))
    
    return lc_messages

# Tool definitions
@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information related to the query."""
    try:
        chunks = vector_indexer.search_similar_chunks(
            query=query,
            project_id="default",
            top_k=3
        )
        
        if chunks:
            return "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunks)])
        else:
            return "No relevant information found in the knowledge base."
    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}")
        return f"Error searching knowledge base: {str(e)}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        # Clean the expression to make it safer
        clean_expr = re.sub(r'[^0-9+\-*/().%\s]', '', expression)
        result = eval(clean_expr)
        return f"The result of {clean_expr} is {result}"
    except Exception as e:
        logger.error(f"Error calculating expression: {e}")
        return f"Error calculating expression: {str(e)}"

@tool
def summarize_conversation(history: str) -> str:
    """Summarize the conversation history."""
    try:
        prompt = ChatPromptTemplate.from_template(SUMMARIZER_PROMPT)
        chain = prompt | llm | StrOutputParser()
        summary = chain.invoke({"history": history})
        return summary
    except Exception as e:
        logger.error(f"Error summarizing conversation: {e}")
        return "Error summarizing conversation."

# Graph nodes
def route_message(state: AgentState) -> AgentState:
    """Determine how to route the user's message"""
    state = log_step(state, "🧭 Routing message...")
    
    try:
        if not state["messages"]:
            return log_step(state, "⚠️ No messages to process")
            
        last_message = state["messages"][-1]["content"]
        history = state.get("history", "")
        
        # Use the router prompt to determine the route
        prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT)
        chain = prompt | llm | StrOutputParser()
        route = chain.invoke({
            "question": last_message,
            "history": history
        }).strip().lower()
        
        # Validate route
        valid_routes = ["rag", "calculator", "direct"]
        if route not in valid_routes:
            route = "direct"  # Default to direct if invalid
        
        state = log_step(state, f"🔀 Routed to: {route}")
        return {**state, "route": route}
        
    except Exception as e:
        error_msg = f"Error routing message: {str(e)}"
        logger.error(error_msg)
        state = log_step(state, f"❌ {error_msg}")
        return {**state, "route": "direct"}  # Default to direct on error

def retrieve_context(state: AgentState) -> AgentState:
    """Retrieve relevant context using RAG"""
    state = log_step(state, "🔍 Searching knowledge base...")
    
    try:
        if not state["messages"]:
            return log_step(state, "⚠️ No messages to process")
            
        last_message = state["messages"][-1]["content"]
        
        # Get relevant chunks from Redis vector store
        chunks = vector_indexer.search_similar_chunks(
            query=last_message,
            project_id="default",
            top_k=3
        )
        
        if chunks:
            context = "\n\n".join([f"📄 Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunks)])
            state = log_step(state, f"✅ Found {len(chunks)} relevant chunks")
        else:
            state = log_step(state, "ℹ️ No specific context found, using general knowledge")
            context = "No specific context found in knowledge base. Using general knowledge."
            
        return {**state, "context": context}
        
    except Exception as e:
        error_msg = f"Error retrieving context: {str(e)}"
        logger.error(error_msg)
        state = log_step(state, f"❌ {error_msg}")
        return {**state, "context": "Error retrieving context. Using general knowledge."}

def perform_calculation(state: AgentState) -> AgentState:
    """Perform calculation for math queries"""
    state = log_step(state, "🧮 Performing calculation...")
    
    try:
        if not state["messages"]:
            return log_step(state, "⚠️ No messages to process")
            
        last_message = state["messages"][-1]["content"]
        
        # Use the calculator prompt
        prompt = ChatPromptTemplate.from_template(CALCULATOR_PROMPT)
        chain = prompt | llm | StrOutputParser()
        calculation_result = chain.invoke({"question": last_message})
        
        state = log_step(state, "✅ Calculation completed")
        
        # Store the result in context
        return {**state, "context": f"Calculation result:\n{calculation_result}"}
        
    except Exception as e:
        error_msg = f"Error performing calculation: {str(e)}"
        logger.error(error_msg)
        state = log_step(state, f"❌ {error_msg}")
        return {**state, "context": "Error performing calculation."}

def update_memory(state: AgentState) -> AgentState:
    """Update short-term and long-term memory"""
    state = log_step(state, "💾 Updating memory...")
    
    try:
        # Update short-term memory with the latest messages
        short_term = state.get("short_term_memory", [])
        messages = state.get("messages", [])
        
        if messages:
            # Add new messages to short-term memory
            for msg in messages:
                if msg not in short_term:
                    short_term.append(msg)
            
            # Keep only the last 10 messages in short-term memory
            short_term = short_term[-10:]
            
            # Update long-term memory if needed
            if len(short_term) >= 5:
                # Summarize conversation for long-term memory
                history_text = format_history(short_term)
                summary = summarize_conversation(history_text)
                
                long_term = state.get("long_term_memory", [])
                long_term.append({
                    "role": "system",
                    "content": f"Conversation summary: {summary}",
                    "timestamp": datetime.now().isoformat()
                })
                
                # Keep only the last 5 summaries in long-term memory
                long_term = long_term[-5:]
                
                state = log_step(state, "✅ Updated long-term memory with summary")
                state = {**state, "long_term_memory": long_term, "summary": summary}
            
            state = log_step(state, f"✅ Updated short-term memory with {len(messages)} messages")
            state = {**state, "short_term_memory": short_term}
        
        return state
        
    except Exception as e:
        error_msg = f"Error updating memory: {str(e)}"
        logger.error(error_msg)
        state = log_step(state, f"❌ {error_msg}")
        return state

def generate_response(state: AgentState) -> AgentState:
    """Generate response using LLM with context and history"""
    state = log_step(state, "🧠 Generating response...")
    
    try:
        if not state["messages"]:
            return log_step(state, "⚠️ No messages to process")
            
        last_message = state["messages"][-1]
        
        # Combine short-term and long-term memory for history context
        short_term = state.get("short_term_memory", [])
        long_term = state.get("long_term_memory", [])
        
        # Format history from memory
        history_text = ""
        
        # Add long-term memory summaries if available
        if long_term:
            history_text += "Previous conversation summaries:\n"
            history_text += "\n".join([msg["content"] for msg in long_term])
            history_text += "\n\nRecent conversation:\n"
        
        # Add short-term memory
        history_text += format_history(short_term)
        
        # Prepare the prompt with context and history
        prompt = SYSTEM_PROMPT.format(
            context=state.get("context", "No specific context available."),
            history=history_text,
            question=last_message["content"]
        )
        
        # Create messages for the LLM
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=last_message["content"])
        ]
        
        # Generate response
        response = llm.invoke(messages)
        state = log_step(state, "✅ Response generated")
        
        # Add assistant's response to messages
        return {
            **state,
            "messages": [*state["messages"], {"role": "assistant", "content": response.content}]
        }
        
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        logger.error(error_msg)
        state = log_step(state, f"❌ {error_msg}")
        return {
            **state,
            "messages": [*state["messages"], {
                "role": "assistant", 
                "content": "I encountered an error. Let me try that again."
            }]
        }

def should_use_tools(state: AgentState) -> Literal["tools", "no_tools"]:
    """Decide whether to use tools based on the route"""
    route = state.get("route", "direct")
    
    if route == "rag":
        return "tools"
    elif route == "calculator":
        return "tools"
    else:
        return "no_tools"

# Define the agent nodes
def create_agent_workflow():
    """Create and return a compiled agent workflow"""
    # Use the initialized memory
    
    # Create the workflow
    workflow = StateGraph(AgentState)
    # Configure workflow to use the memory saver
    workflow.set_checkpointer(memory)
    
    # Add nodes
    workflow.add_node("router", route_message)
    workflow.add_node("retrieve", retrieve_context)
    workflow.add_node("calculate", perform_calculation)
    workflow.add_node("memory", update_memory)
    workflow.add_node("generate", generate_response)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "router",
        should_use_tools,
        {
            "tools": "retrieve",
            "no_tools": "memory"
        }
    )
    
    # Add the rest of the edges
    workflow.add_edge("retrieve", "memory")
    workflow.add_edge("calculate", "memory")
    workflow.add_edge("memory", "generate")
    workflow.add_edge("generate", END)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Compile the workflow
    return workflow.compile()

def check_vector_store():
    """Check if the vector store is properly initialized"""
    try:
        health = vector_indexer.check_index_health("default")
        logger.info(f"Vector store health check: {health}")
        return health
    except Exception as e:
        logger.error(f"Error checking vector store: {e}")
        return {"error": str(e)}

# Initialize the agent
agent = create_agent_workflow()

# Check vector store on startup
check_vector_store()

def create_conversation_thread() -> str:
    """
    Create a new conversation thread.
    
    Returns:
        Thread ID as a string
    """
    thread_id = str(uuid.uuid4())
    logger.info(f"Created new conversation thread: {thread_id}")
    return thread_id

async def process_message(thread_id: str, content: str, quote: str = None) -> tuple[str, list]:
    """
    Process a message through the agent with RAG and conversation history
    
    Args:
        thread_id: The conversation thread ID
        content: The message content
        quote: Optional quoted text from the conversation
        
    Returns:
        A tuple of (response_text, thinking_steps)
    """
    try:
        # Get conversation history
        history_messages = get_conversation_history(thread_id)
        history_text = format_history(history_messages[-5:])  # Last 5 messages
        
        # Prepare the message with quote if provided
        user_message = f"{quote}\n\n{content}" if quote else content
        
        # Initialize state with messages and history
        state = {
            "messages": [{"role": "user", "content": user_message}],
            "context": "",
            "thinking_steps": [],
            "history": history_text,
            "thread_id": thread_id,
            "short_term_memory": history_messages[-10:] if history_messages else [],
            "long_term_memory": [],
            "working_memory": {},
            "route": "direct",
            "tool_calls": [],
            "tool_results": [],
            "summary": ""
        }
        
        # Run the agent
        result = agent.invoke(state)
        
        # Get the assistant's response and thinking steps
        assistant_response = result["messages"][-1]["content"]
        thinking_steps = result.get("thinking_steps", [])
        
        # Update conversation history
        update_conversation_history(thread_id, "user", content)
        update_conversation_history(thread_id, "assistant", assistant_response)
        
        return assistant_response, thinking_steps
        
    except Exception as e:
        error_msg = f"Error in process_message: {str(e)}"
        logger.error(error_msg)
        return "I encountered an error processing your message. Please try again.", [error_msg]
