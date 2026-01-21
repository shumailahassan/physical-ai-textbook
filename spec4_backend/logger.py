"""
Standalone logging module for the RAG agent system.

This module logs:
- Timestamp
- Incoming query
- Agent response
Using standard Python logging.
"""
import logging
from datetime import datetime
from typing import Optional


class RequestResponseLogger:
    """
    Logger class for tracking requests and responses.
    """

    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize the logger.

        Args:
            log_file: Optional file to log to (if None, logs to console only)
        """
        # Setup basic logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        self.logger = logging.getLogger(__name__)

        # If a log file is specified, add a file handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def log_request(self, query: str) -> str:
        """
        Log an incoming query.

        Args:
            query: The incoming query string

        Returns:
            Timestamp string for correlation
        """
        timestamp = datetime.now().isoformat()
        self.logger.info(f"REQUEST - Timestamp: {timestamp} - Query: {query}")
        return timestamp

    def log_response(self, query: str, response: str, timestamp: str) -> None:
        """
        Log an agent response.

        Args:
            query: The original query
            response: The agent's response
            timestamp: The timestamp from when the request was logged
        """
        response_timestamp = datetime.now().isoformat()
        self.logger.info(f"RESPONSE - Request_Timestamp: {timestamp} - Response_Timestamp: {response_timestamp} - Query: {query} - Response_Length: {len(response)}")


# Global logger instance
request_response_logger = RequestResponseLogger()


def log_request(query: str) -> str:
    """
    Log an incoming query.

    Args:
        query: The incoming query string

    Returns:
        Timestamp string for correlation
    """
    return request_response_logger.log_request(query)


def log_response(query: str, response: str, timestamp: str) -> None:
    """
    Log an agent response.

    Args:
        query: The original query
        response: The agent's response
        timestamp: The timestamp from when the request was logged
    """
    request_response_logger.log_response(query, response, timestamp)