"""
Initialization module for spec4_backend package.
Exports key components for the FastAPI application.
"""

__version__ = "1.0.0"

# Define what gets imported with "from spec4_backend import *"
__all__ = ['app']

# Import the FastAPI app for easy access if needed
try:
    from .api import app
except ImportError:
    # The app might not be importable if dependencies are missing
    app = None