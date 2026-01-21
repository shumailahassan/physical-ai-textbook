"""
Spec-3 Agent Integration Bridge

This module provides an interface to the Spec-3 agent modules
for use by the FastAPI application.
"""
import sys
import os
from typing import Dict, Any
import logging

# Add parent directory to path to import from spec3_backend
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

# Import Spec-3 modules
from spec3_backend.query_handler import QueryHandler
from spec3_backend.agent_init import OpenAIAgent
from spec3_backend.retrieval_integration import RetrievalIntegration

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentBridge:
    """
    Bridge class to connect with Spec-3 agent modules.
    """

    def __init__(self):
        """
        Initialize the agent bridge by setting up Spec-3 components.
        """
        logger.info("Initializing Agent Bridge with Spec-3 modules")

        try:
            # Initialize the query handler which connects to agent and retrieval
            self.query_handler = QueryHandler()
            logger.info("Query handler initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize agent bridge: {e}")
            raise


    def process_query_via_agent(self, query: str) -> Dict[str, Any]:
        """
        Process a query using the Spec-3 agent pipeline.

        Args:
            query: The user query to process

        Returns:
            Dictionary containing the agent's response and metadata
        """
        logger.info(f"Processing query via agent: {query[:50]}...")

        try:
            # Use the query handler to process the query through the agent pipeline
            result = self.query_handler.process_query(query)

            logger.info(f"Query processed successfully, response length: {len(result.get('response', ''))}")
            return result

        except Exception as e:
            logger.error(f"Error processing query via agent: {e}")
            raise


# Global instance of the agent bridge
agent_bridge = AgentBridge()


def process_query(query: str) -> Dict[str, Any]:
    """
    Process a query using the Spec-3 agent pipeline.
    This is a convenience function that uses the global agent bridge instance.

    Args:
        query: The user query to process

    Returns:
        Dictionary containing the agent's response and metadata
    """
    return agent_bridge.process_query_via_agent(query)