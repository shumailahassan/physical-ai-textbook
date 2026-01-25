# RAG-Based Chatbot Backend

This is a complete backend for a RAG (Retrieval-Augmented Generation) based chatbot system that integrates OpenAI/Claude agents with vector database retrieval.

## Project Structure

```
physical-ai-textbook/
├── spec3_backend/              # Core RAG functionality
│   ├── agent_init.py           # OpenAI/Claude Agent initialization
│   ├── retrieval_integration.py # Retrieval logic for vector database
│   ├── query_handler.py        # Handles user queries and integrates Agent + Retrieval
│   └── __init__.py
├── spec4_backend/              # FastAPI application
│   ├── api.py                  # FastAPI app, POST /query endpoint
│   └── __init__.py
├── .env                        # Environment variables (API keys, DB urls, etc.)
└── requirements.txt            # Dependencies
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the `.env.example` (or use the existing `.env`) and fill in your API keys:

```bash
# OpenAI API key (required for OpenAI agent)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API key (optional, for Claude agent)
CLAUDE_API_KEY=your_claude_api_key_here

# Cohere API key (required for embeddings)
COHERE_API_KEY=your_cohere_api_key_here

# Qdrant vector database credentials
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here

# Configuration parameters
VECTOR_SIZE=384                 # Size of embedding vectors
CHUNK_SIZE=500                  # Size of text chunks
OVERLAP=50                      # Overlap between chunks
MAX_PAGES=10                    # Maximum pages to process
TEXTBOOK_URLS=https://example.com/textbook1,https://example.com/textbook2
```

### 3. Run the Application

Start the FastAPI server:

```bash
cd physical-ai-textbook
python -m uvicorn spec4_backend.api:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST /query
Process user queries using the RAG agent.

**Request Body:**
```json
{
  "query": "Your question here"
}
```

**Response:**
```json
{
  "query": "Your question here",
  "response": "Agent's response",
  "timestamp": "2023-10-01T12:00:00.000Z"
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "RAG-Based Chatbot API",
  "timestamp": "2023-10-01T12:00:00.000Z"
}
```

## Testing the API

After starting the server, you can test the API using curl:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the principles of artificial intelligence in robotics?"}'
```

Or using Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"query": "What are the principles of artificial intelligence in robotics?"}
)

print(response.json())
```

## Components

### Agent Initialization (`spec3_backend/agent_init.py`)
- Initializes OpenAI or Claude agents
- Handles API key management
- Provides testing functionality

### Retrieval Integration (`spec3_backend/retrieval_integration.py`)
- Integrates with Cohere for embeddings
- Connects to Qdrant for vector storage
- Implements similarity search

### Query Handler (`spec3_backend/query_handler.py`)
- Coordinates retrieval and generation
- Formats context for the agent
- Processes user queries using RAG methodology

### FastAPI Application (`spec4_backend/api.py`)
- Provides REST API endpoints
- Handles request/response formatting
- Implements health checks

## Troubleshooting

1. **API Keys**: Ensure all required API keys are set in the `.env` file
2. **Dependencies**: Make sure all packages in `requirements.txt` are installed
3. **Vector Database**: Ensure Qdrant is accessible if using real retrieval (mock data used otherwise)

## Development

To run tests:
```bash
# Run individual modules to test functionality
python -m spec3_backend.agent_init
python -m spec3_backend.retrieval_integration
python -m spec3_backend.query_handler
```
