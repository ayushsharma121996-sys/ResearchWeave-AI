"""
Vercel serverless entrypoint for ResearchWeave AI (FastAPI).
Imports the core app object from app/api.py.
"""
from app.api import app

# Export application for Vercel
__all__ = ["app"]
