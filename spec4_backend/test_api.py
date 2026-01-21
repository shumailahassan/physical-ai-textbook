#!/usr/bin/env python3
"""
Test script to verify the API endpoints are working correctly.
"""

import asyncio
import json
import sys
import os

# Add parent directory to path to import from spec3_backend
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from api import app
from api import ChatRequest
from spec3_backend.query_handler import create_query_handler

async def test_chat_endpoint():
    """Test the chat endpoint manually."""
    print("Testing Claude agent initialization...")

    try:
        # Test creating a query handler
        query_handler = create_query_handler(agent_type='claude')
        print("SUCCESS: Query handler created successfully")

        # Test a simple query
        result = query_handler.process_query("Hello, are you working?")
        print("SUCCESS: Query processed successfully")
        print(f"Response: {result.get('response', 'No response field')[:100]}...")

    except Exception as e:
        print(f"ERROR: Error testing query handler: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    print("Testing API functionality...")
    success = asyncio.run(test_chat_endpoint())

    if success:
        print("\nSUCCESS: API test passed! The backend should work correctly.")
    else:
        print("\nERROR: API test failed. Check the errors above.")