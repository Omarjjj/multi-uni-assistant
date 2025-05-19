#!/usr/bin/env python
"""
Script to run the Multi-University Assistant backend server.
"""

import uvicorn
import os
import sys
from dotenv import load_dotenv

def check_env():
    """Check if environment variables are set."""
    load_dotenv()
    
    required_vars = [
        "OPENAI_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_ENV",
        "PINECONE_INDEX"
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print("❌ Error: Missing environment variables:")
        for var in missing:
            print(f"  - {var}")
        print("\nPlease create a .env file with the missing variables.")
        sys.exit(1)
    
    print("✅ Environment variables loaded successfully")

if __name__ == "__main__":
    # Port can be overridden with command line argument
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    
    print("🔍 Checking environment variables...")
    check_env()
    
    print(f"🚀 Starting server on http://localhost:{port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True) 