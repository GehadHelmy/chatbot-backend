# main.py - FastAPI app for Railway Deployment
import os
import sys
from datetime import datetime, date
from typing import Optional, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# FastAPI + external libs
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, validator
from supabase import create_client, Client
from langchain_core.prompts import PromptTemplate
from langchain_cohere import ChatCohere
import hashlib
import jwt
from collections import Counter
import traceback

# -------------------------
# 🔧 Environment Variables
# -------------------------
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
JWT_SECRET = os.getenv("JWT_SECRET", "default-secret")
DEBUG_MODE = os.getenv("DEBUG", "False").lower() == "true"

# Set Cohere API key
if COHERE_API_KEY:
    os.environ["COHERE_API_KEY"] = COHERE_API_KEY

# -------------------------
# 🚀 FastAPI App
# -------------------------
app = FastAPI(
    title="InsideOut Chatbot API",
    version="1.0.0",
    description="An empathetic AI chatbot for emotional support",
    debug=DEBUG_MODE
)

# -------------------------
# 🌐 CORS
# -------------------------
allowed_origins = ["*"]  # Railway + Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# 🗃️ Database
# -------------------------
supabase: Optional[Client] = None

def initialize_database():
    global supabase
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        supabase.table("users").select("id").limit(1).execute()
        logger.info("✅ Supabase connected")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

initialize_database()

# -------------------------
# 🔐 Security
# -------------------------
security = HTTPBearer(auto_error=False)

def create_access_token(user_id: str, username: str) -> str:
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": datetime.utcnow().timestamp() + 86400
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def verify_token(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Authorization header required")
    return verify_token(credentials.credentials)

# -------------------------
# Routes
# -------------------------
@app.get("/")
async def root():
    return {
        "success": True,
        "message": "InsideOut Chatbot API is running on Railway!",
        "database_connected": supabase is not None,
        "ai_available": COHERE_API_KEY is not None
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "database_connected": supabase is not None,
        "ai_available": COHERE_API_KEY is not None,
        "time": datetime.utcnow().isoformat()
    }

@app.get("/test")
async def test():
    return HTMLResponse("<h1>Railway API works 🎉</h1>")

# -------------------------
# Run Uvicorn on Railway
# -------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
