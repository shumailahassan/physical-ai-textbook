"""
FastAPI application for the RAG-based chatbot system.
Provides endpoints for querying the RAG agent and health checks.
"""

import os
from dotenv import load_dotenv
import sys
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List

# ======================
# Load environment variables FIRST
# ======================
load_dotenv()
print("OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))  # Optional: verify key

# ======================
# Add project root to path for imports
# ======================
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add parent directory to import from spec3_backend
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ======================
# Import QueryHandler from spec3_backend
# ======================
try:
    from spec3_backend.query_handler import QueryHandler, create_query_handler
    print("DEBUG: Successfully imported QueryHandler")
except Exception as e:
    print(f"DEBUG: Failed to import QueryHandler: {e}")
    import traceback
    traceback.print_exc()
    raise

# ======================
# Logging setup
# ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================
# Initialize FastAPI app
# ======================
app = FastAPI(
    title="RAG-Based Chatbot API",
    description="API for RAG-based chatbot system with OpenAI/Claude integration",
    version="1.0.0"
)

# ======================
# Add CORS middleware
# ======================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins during development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
    # Expose headers can be added if needed
)

# ======================
# Define request/response models
# ======================
class AskRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = []

class AskResponse(BaseModel):
    response: str

# ======================
# Request and Response Models
# ======================
class QueryRequest(BaseModel):
    query: str

class ChatRequest(BaseModel):
    message: str
    history: list = []

class QueryResponse(BaseModel):
    query: str
    response: str
    timestamp: str

class ChatResponse(BaseModel):
    response: str

# ======================
# Ask Endpoint (for frontend compatibility)
# ======================
@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: AskRequest) -> Dict[str, str]:
    logger.info(f"Received ask request: '{request.query[:50]}...'")
    try:
        # Create QueryHandler with Claude agent
        query_handler = create_query_handler(agent_type='claude')

        # Process query using QueryHandler
        result = query_handler.process_query(request.query)

        if 'error' in result:
            logger.error(f"Error in query processing: {result['error']}")
            raise HTTPException(status_code=500, detail=result['error'])

        response = AskResponse(
            response=result['response']
        )

        logger.info(f"Ask request processed successfully. Response length: {len(result['response'])} chars")
        return {"response": result['response']}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing ask request: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# ======================
# Query Endpoint
# ======================
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> Dict[str, Any]:
    logger.info(f"Received query: '{request.query[:50]}...'")
    try:
        # Create QueryHandler with Claude agent
        query_handler = create_query_handler(agent_type='claude')

        # Process query
        result = query_handler.process_query(request.query)

        if 'error' in result:
            logger.error(f"Error in query processing: {result['error']}")
            raise HTTPException(status_code=500, detail=result['error'])

        response = QueryResponse(
            query=result['query'],
            response=result['response'],
            timestamp=result['timestamp']
        )

        logger.info(f"Query processed successfully. Response length: {len(result['response'])} chars")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# ======================
# Simple test endpoint
# ======================
@app.post("/test")
async def test_endpoint(request: dict):
    return {"received": request, "message": "Test successful"}

# ======================
# Chat Endpoint (for frontend compatibility)
# ======================
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    print(f"DEBUG: Chat endpoint called with request: {request}")
    logger.info(f"Received chat message: '{request.message[:50]}...'")
    try:
        # Create QueryHandler with Claude agent
        print("DEBUG: Creating query handler...")
        import os
        print(f"DEBUG: ANTHROPIC_BASE_URL env var: {os.getenv('ANTHROPIC_BASE_URL')}")
        print(f"DEBUG: CLAUDE_API_KEY env var exists: {bool(os.getenv('CLAUDE_API_KEY'))}")

        query_handler = create_query_handler(agent_type='claude')
        print("DEBUG: Query handler created successfully")

        # Process the message (ignore history for now, can be extended later)
        print(f"DEBUG: Processing message: {request.message}")
        result = query_handler.process_query(request.message)
        print(f"DEBUG: Got result: {result}")

        # Return a simple dictionary response
        return {"response": result['response']}

    except Exception as e:
        print(f"DEBUG: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"Unexpected error processing chat: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# ======================
# Health Check Endpoint
# ======================
@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {
        "status": "healthy",
        "service": "RAG-Based Chatbot API",
        "timestamp": datetime.utcnow().isoformat()
    }

# ======================
# Run via uvicorn
# ======================
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting RAG-Based Chatbot API server...")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
