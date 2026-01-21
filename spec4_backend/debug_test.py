#!/usr/bin/env python3
"""
Debug test to see what's happening with the API
"""

import sys
import os

# Add parent directory to path to import from spec3_backend
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from spec4_backend.api import ChatRequest, create_query_handler

def test_direct_call():
    print("Testing direct function call...")

    # Create a mock request
    request = ChatRequest(message="Hello, how are you?", history=[])
    print(f"Request created: {request}")

    try:
        # Create QueryHandler with Claude agent
        print("Creating query handler...")
        query_handler = create_query_handler(agent_type='claude')
        print("Query handler created successfully")

        # Process the message (ignore history for now, can be extended later)
        print(f"Processing message: {request.message}")
        result = query_handler.process_query(request.message)
        print(f"Got result: {result}")

        # Create response
        from spec4_backend.api import ChatResponse
        response = ChatResponse(response=result['response'])
        print(f"Final response: {response}")

        return response

    except Exception as e:
        print(f"Error in processing: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_direct_call()