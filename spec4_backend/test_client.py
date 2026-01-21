"""
Minimal test script to simulate a frontend HTTP request to the FastAPI backend.

This script sends a sample query to the FastAPI endpoint and prints the response.
"""
import requests
import json
import sys
from typing import Dict, Any


def test_fastapi_endpoint():
    """
    Simulate a frontend HTTP request to the FastAPI endpoint.
    """
    # Sample query to send
    sample_query = {"query": "Test query for the RAG agent"}

    print("=== Frontend-Backend Test ===")
    print("Sending test query to FastAPI endpoint...")
    print(f"Query: {sample_query['query']}")

    try:
        # Send POST request to the query endpoint
        # Note: This assumes the FastAPI server is running on localhost:8000
        response = requests.post("http://localhost:8000/query", json=sample_query)

        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            response_data = response.json()
            print(f"Response content: {json.dumps(response_data, indent=2)}")

            print("\nParsed Response:")
            print(f"Original Query: {response_data.get('query', 'N/A')}")
            print(f"Agent Response: {response_data.get('response', 'N/A')[:100]}...")
            print(f"Timestamp: {response_data.get('timestamp', 'N/A')}")
            print("\n✓ Test completed successfully!")
            return True
        else:
            print(f"✗ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("✗ Connection error: Could not connect to FastAPI server")
        print("Make sure the FastAPI server is running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def main():
    """
    Main function to run the test.
    """
    print("Frontend-Backend Test Script")
    print("This script simulates a frontend HTTP request to the FastAPI backend.")
    print("Note: Make sure the FastAPI server is running on http://localhost:8000 before running this test.\n")

    success = test_fastapi_endpoint()

    if success:
        print("\n✓ Backend is ready for local frontend integration!")
    else:
        print("\n✗ Backend test failed. Please ensure the FastAPI server is running.")
        sys.exit(1)


if __name__ == "__main__":
    main()