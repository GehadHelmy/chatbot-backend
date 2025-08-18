# main.py - Production-ready FastAPI for PythonAnywhere
import os
import sys
from datetime import datetime, date
from typing import Optional, List, Dict, Any
import logging

# Configure logging for PythonAnywhere
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/insideout_api.log') if os.path.exists('/tmp') else logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add current directory to Python path for PythonAnywhere
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from fastapi import FastAPI, HTTPException, Depends, status, Request
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, EmailStr, validator
    from supabase import create_client, Client
    from langchain_core.prompts import PromptTemplate
    from langchain_cohere import ChatCohere
    import hashlib
    import secrets
    import jwt
    from collections import Counter
    import traceback
    from dotenv import load_dotenv
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

# Load environment variables
load_dotenv()

# -------------------------
# 🔧 Configuration
# -------------------------

# Environment Variables - No hardcoded defaults for security
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
JWT_SECRET = os.getenv("JWT_SECRET")

# Platform specific configurations
IS_PYTHONANYWHERE = os.getenv("PYTHONANYWHERE_SITE", "").startswith("www.pythonanywhere.com")
IS_RENDER = os.getenv("RENDER", "False").lower() == "true"
DEBUG_MODE = os.getenv("DEBUG", "False").lower() == "true"

# Set Cohere API key in environment
if COHERE_API_KEY:
    os.environ["COHERE_API_KEY"] = COHERE_API_KEY

# FastAPI App Configuration
app = FastAPI(
    title="InsideOut Chatbot API",
    version="1.0.0",
    description="An empathetic AI chatbot for emotion support",
    docs_url="/docs" if DEBUG_MODE else None,  # Disable docs in production on PythonAnywhere
    redoc_url="/redoc" if DEBUG_MODE else None,
    debug=DEBUG_MODE
)

# -------------------------
# 🌐 CORS Configuration for Flutter
# -------------------------
# More restrictive CORS for production
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:3000",
    "file://",  # Allow local file access
    "null",     # Allow local file access
    "https://your-flutter-app-domain.com",  # Replace with your actual domain
]

# For local development and Render, allow all origins
if not IS_PYTHONANYWHERE or IS_RENDER:
    allowed_origins = ["*"]

if DEBUG_MODE:
    allowed_origins.append("*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False if not IS_PYTHONANYWHERE else True,  # Disable credentials for local dev
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# -------------------------
# 📁 Static Files (for PythonAnywhere)
# -------------------------
try:
    # Mount static files if they exist
    static_dir = os.path.join(current_dir, "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        logger.info("Static files mounted at /static")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

# -------------------------
# 🗃️ Database Connection
# -------------------------
supabase: Optional[Client] = None

def initialize_database():
    """Initialize database connection with retry logic"""
    global supabase
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            # Test the connection
            result = supabase.table("users").select("id").limit(1).execute()
            logger.info("Supabase connection established successfully")
            return True
        except Exception as e:
            logger.error(f"Database connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error("Failed to establish database connection after all retries")
                return False

# Initialize database on startup
if not initialize_database():
    logger.warning("Database connection failed - some features will be unavailable")

# -------------------------
# 🔐 Security
# -------------------------
security = HTTPBearer(auto_error=False)

# -------------------------
# 📋 Pydantic Models (Flutter Compatible)
# -------------------------

class ApiResponse(BaseModel):
    """Standardized API response for Flutter"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str = datetime.utcnow().isoformat()

class UserRegister(BaseModel):
    username: str
    email: EmailStr
    
    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3 or len(v) > 50:
            raise ValueError('Username must be between 3 and 50 characters')
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username can only contain letters, numbers, hyphens, and underscores')
        return v.lower()

class UserLogin(BaseModel):
    username: str
    
    @validator('username')
    def validate_username(cls, v):
        return v.lower()

class ChatRequest(BaseModel):
    message: str
    
    @validator('message')
    def validate_message(cls, v):
        if len(v.strip()) == 0:
            raise ValueError('Message cannot be empty')
        if len(v) > 1000:
            raise ValueError('Message too long (max 1000 characters)')
        return v.strip()

class MoodEntry(BaseModel):
    emotion: str
    character_name: str
    led_color: str
    sound_file: str
    
    @validator('emotion')
    def validate_emotion(cls, v):
        valid_emotions = ['joy', 'sadness', 'anger', 'fear', 'disgust', 'neutral']
        if v.lower() not in valid_emotions:
            raise ValueError(f'Emotion must be one of: {", ".join(valid_emotions)}')
        return v.lower()

class UserProfile(BaseModel):
    id: str
    username: str
    email: str
    created_at: str

class AuthResponse(BaseModel):
    success: bool
    message: str
    token: Optional[str] = None
    user: Optional[UserProfile] = None
    error: Optional[str] = None
    timestamp: str = datetime.utcnow().isoformat()

# -------------------------
# 🤖 AI Configuration
# -------------------------

# Initialize AI models with error handling
AI_AVAILABLE = False
llm = None
emotion_chain = None
response_chain = None

def initialize_ai():
    """Initialize AI models with retry logic"""
    global AI_AVAILABLE, llm, emotion_chain, response_chain
    
    if not COHERE_API_KEY:
        logger.warning("Cohere API key not provided - AI features disabled")
        return False
    
    try:
        llm = ChatCohere(model="command-r-plus", temperature=0)
        
        emotion_prompt = PromptTemplate.from_template("""
You are an expert emotion classifier. Take a look at the message, If the message is completely neutral and has no emotional tone at all,
respond with: No Emotion.
Else if the emotion is unclear but has any emotional undertone, choose the closest match out of exactly one of these:
joy, sadness, anger, fear, disgust.
Message: {message}

Respond with one word only: joy, sadness, anger, fear, disgust, or No Emotion.
""")

        response_prompt = PromptTemplate.from_template("""
        You are a warm, deeply empathetic assistant who truly listens to the user as a compassionate human friend. 
        Your goal is to understand, reflect, and respond in ways that leave the user feeling heard, valued, and supported.

        User's recent context (for your understanding only):
        {history}

        Current message: {message}
        Detected emotion: {emotion}

        Response guidelines:

        1. **Empathy and Validation:**
           - Acknowledge their emotion without judgment
           - Use warm, natural language - avoid being robotic
           - Celebrate improvements, offer comfort for struggles
           - Share in their joy when they're happy

        2. **For neutral emotions:**
           - Be friendly and engaging
           - Gently invite them to share their feelings

        3. **Crisis Response (CRITICAL):**
           - If ANY signs of suicidal ideation, self-harm, or severe hopelessness:
           - Respond with deep care and seriousness
           - Validate their pain and remind them their life matters
           - Ask for their location to provide relevant crisis resources
           - Provide multiple international crisis hotlines
           - Encourage immediate professional support
           - Give a warm, detailed response - never be dismissive

        4. **Tone:**
           - Speak as a caring human friend
           - Use encouraging phrases like "I'm here with you"
           - Avoid clinical or formal language

        Keep responses under 200 words but ensure warmth and completeness.
        """)

        emotion_chain = emotion_prompt | llm
        response_chain = response_prompt | llm
        AI_AVAILABLE = True
        logger.info("AI models initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize AI models: {e}")
        AI_AVAILABLE = False
        return False

# Initialize AI on startup
initialize_ai()

# -------------------------
# 🔐 Authentication Functions
# -------------------------

def hash_password(password: str) -> str:
    """Hash password using SHA-256 (in production, use bcrypt)"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_access_token(user_id: str, username: str) -> str:
    """Create JWT access token"""
    try:
        payload = {
            "user_id": user_id,
            "username": username,
            "exp": datetime.utcnow().timestamp() + 86400,  # 24 hours
            "iat": datetime.utcnow().timestamp()
        }
        return jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    except Exception as e:
        logger.error(f"Token creation failed: {e}")
        raise HTTPException(status_code=500, detail="Token creation failed")

def verify_token(token: str) -> dict:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        if payload["exp"] < datetime.utcnow().timestamp():
            raise HTTPException(status_code=401, detail="Token expired")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    payload = verify_token(credentials.credentials)
    return payload

# -------------------------
# 🗃️ Database Functions with Error Handling
# -------------------------

def safe_db_operation(operation_name: str, operation_func):
    """Wrapper for safe database operations"""
    try:
        if not supabase:
            raise Exception("Database connection not available")
        return operation_func()
    except Exception as e:
        logger.error(f"{operation_name} failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database operation failed: {operation_name}")

def save_emotion(user_id: str, emotion: str, message: str) -> bool:
    """Save emotion log to database"""
    def operation():
        result = supabase.table("emotion_logs").insert({
            "user_id": user_id,
            "emotion": emotion,
            "message": message,
            "created_at": datetime.utcnow().isoformat()
        }).execute()
        return result.data is not None
    
    try:
        return safe_db_operation("save_emotion", operation)
    except:
        logger.error(f"Failed to save emotion for user {user_id}")
        return False

def get_user_history(user_id: str, limit: int = 5) -> str:
    """Get user's chat history"""
    def operation():
        result = supabase.table("emotion_logs") \
            .select("created_at, emotion, message") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        
        rows = result.data
        if not rows:
            return "No prior messages."
        
        history_lines = []
        for row in reversed(rows):
            timestamp = row['created_at'][:19]  # Remove microseconds
            history_lines.append(f"{timestamp}: ({row['emotion']}) {row['message'][:100]}")
        
        return "\n".join(history_lines)
    
    try:
        return safe_db_operation("get_user_history", operation)
    except:
        return "No prior messages available."

def get_daily_emotions(user_id: str, target_date: str = None) -> Dict[str, Any]:
    """Get emotions for a specific day"""
    if not target_date:
        target_date = date.today().isoformat()
    
    def operation():
        result = supabase.table("emotion_logs") \
            .select("emotion, created_at") \
            .eq("user_id", user_id) \
            .gte("created_at", f"{target_date}T00:00:00") \
            .lt("created_at", f"{target_date}T23:59:59") \
            .execute()
        
        emotions = [row['emotion'] for row in result.data if row['emotion'] not in ['No Emotion', 'neutral']]
        emotion_counts = Counter(emotions)
        
        return {
            "date": target_date,
            "emotions": dict(emotion_counts),
            "total_messages": len(result.data),
            "dominant_emotion": emotion_counts.most_common(1)[0][0] if emotion_counts else "neutral"
        }
    
    try:
        return safe_db_operation("get_daily_emotions", operation)
    except:
        return {
            "date": target_date,
            "emotions": {},
            "total_messages": 0,
            "dominant_emotion": "neutral"
        }

def get_mood_history(user_id: str, days: int = 7) -> List[Dict[str, Any]]:
    """Get mood history for specified number of days"""
    mood_data = []
    current_date = date.today()
    
    for i in range(days):
        check_date = date.fromordinal(current_date.toordinal() - i)
        daily_mood = get_daily_emotions(user_id, check_date.isoformat())
        mood_data.append(daily_mood)
    
    return list(reversed(mood_data))

def clear_user_history(user_id: str) -> Dict[str, Any]:
    """Clear user's chat history"""
    def operation():
        result = supabase.table("emotion_logs").delete().eq("user_id", user_id).execute()
        return {"success": True, "message": "Chat history cleared successfully"}
    
    try:
        return safe_db_operation("clear_user_history", operation)
    except:
        return {"success": False, "message": "Failed to clear chat history"}

# -------------------------
# 🚨 Global Exception Handler
# -------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}\nTraceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content=ApiResponse(
            success=False,
            message="Internal server error",
            error="An unexpected error occurred. Please try again later."
        ).dict()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ApiResponse(
            success=False,
            message=str(exc.detail),
            error=str(exc.detail)
        ).dict()
    )

# -------------------------
# 🛣️ API Routes (Flutter Compatible)
# -------------------------

@app.get("/", response_model=ApiResponse)
async def root():
    """Root endpoint - health check"""
    return ApiResponse(
        success=True,
        message="InsideOut Chatbot API is running!",
        data={
            "version": "1.0.0",
            "status": "healthy",
            "ai_available": AI_AVAILABLE,
            "database_connected": supabase is not None,
            "environment": "production" if (IS_PYTHONANYWHERE or IS_RENDER) else "development"
        }
    )

@app.get("/health", response_model=ApiResponse)
async def health_check():
    """Detailed health check for monitoring"""
    health_data = {
        "api_status": "healthy",
        "database_connected": supabase is not None,
        "ai_available": AI_AVAILABLE,
        "timestamp": datetime.utcnow().isoformat(),
        "environment": "production" if (IS_PYTHONANYWHERE or IS_RENDER) else "development"
    }
    
    # Test database connection
    if supabase:
        try:
            result = supabase.table("users").select("id").limit(1).execute()
            health_data["database_test"] = "passed"
        except:
            health_data["database_test"] = "failed"
            health_data["database_connected"] = False
    
    return ApiResponse(
        success=True,
        message="Health check completed",
        data=health_data
    )

@app.post("/register", response_model=AuthResponse)
async def register(user_data: UserRegister):
    """User registration endpoint"""
    if not supabase:
        return AuthResponse(
            success=False,
            message="Registration unavailable",
            error="Database connection not available"
        )
    
    try:
        # Check if username exists
        existing_user = supabase.table("users").select("id").eq("username", user_data.username).execute()
        if existing_user.data:
            return AuthResponse(
                success=False,
                message="Registration failed",
                error="Username already exists"
            )
        
        # Check if email exists
        existing_email = supabase.table("users").select("id").eq("email", user_data.email).execute()
        if existing_email.data:
            return AuthResponse(
                success=False,
                message="Registration failed",
                error="Email already exists"
            )
        
        # Create new user
        result = supabase.table("users").insert({
            "username": user_data.username,
            "email": user_data.email,
            "created_at": datetime.utcnow().isoformat()
        }).execute()
        
        if not result.data:
            raise Exception("User creation failed")
        
        user = result.data[0]
        token = create_access_token(user["id"], user["username"])
        
        return AuthResponse(
            success=True,
            message="User registered successfully",
            token=token,
            user=UserProfile(
                id=user["id"],
                username=user["username"],
                email=user["email"],
                created_at=user["created_at"]
            )
        )
        
    except ValueError as e:
        return AuthResponse(
            success=False,
            message="Registration failed",
            error=str(e)
        )
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return AuthResponse(
            success=False,
            message="Registration failed",
            error="An error occurred during registration"
        )

@app.post("/login", response_model=AuthResponse)
async def login(user_data: UserLogin):
    """User login endpoint"""
    if not supabase:
        return AuthResponse(
            success=False,
            message="Login unavailable",
            error="Database connection not available"
        )
    
    try:
        result = supabase.table("users").select("*").eq("username", user_data.username).execute()
        
        if not result.data:
            return AuthResponse(
                success=False,
                message="Login failed",
                error="Invalid username"
            )
        
        user = result.data[0]
        token = create_access_token(user["id"], user["username"])
        
        return AuthResponse(
            success=True,
            message="Login successful",
            token=token,
            user=UserProfile(
                id=user["id"],
                username=user["username"],
                email=user["email"],
                created_at=user["created_at"]
            )
        )
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return AuthResponse(
            success=False,
            message="Login failed",
            error="An error occurred during login"
        )

@app.get("/start", response_model=ApiResponse)
async def start_conversation(current_user: dict = Depends(get_current_user)):
    """Start a new conversation"""
    return ApiResponse(
        success=True,
        message="Conversation started",
        data={
            "emotion": "neutral",
            "reply": f"Hi {current_user['username']}, I'm here for you. How are you feeling today?",
            "user": current_user['username']
        }
    )

@app.post("/chat", response_model=ApiResponse)
async def chat_endpoint(req: ChatRequest, current_user: dict = Depends(get_current_user)):
    """Main chat endpoint"""
    if not AI_AVAILABLE:
        return ApiResponse(
            success=False,
            message="Chat unavailable",
            error="AI service is not available"
        )
    
    try:
        user_id = current_user["user_id"]
        
        # Get emotion detection
        try:
            emotion_result = emotion_chain.invoke({"message": req.message})
            detected_emotion = emotion_result.content.strip().lower()
            
            # Normalize emotion
            if detected_emotion not in ['joy', 'sadness', 'anger', 'fear', 'disgust']:
                detected_emotion = 'neutral'
                
        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            detected_emotion = 'neutral'
        
        # Save to database (non-blocking)
        save_emotion(user_id, detected_emotion, req.message)
        
        # Get user history
        history = get_user_history(user_id)
        
        # Generate response
        try:
            bot_response = response_chain.invoke({
                "emotion": detected_emotion,
                "message": req.message,
                "history": history
            })
            reply = bot_response.content.strip()
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            reply = "I'm here to listen and support you. Could you tell me more about how you're feeling?"
        
        return ApiResponse(
            success=True,
            message="Chat response generated",
            data={
                "emotion": detected_emotion,
                "reply": reply
            }
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ApiResponse(
            success=False,
            message="Chat processing failed",
            error="Unable to process your message right now"
        )

@app.delete("/clear-history", response_model=ApiResponse)
async def clear_history(current_user: dict = Depends(get_current_user)):
    """Clear user's chat history"""
    user_id = current_user["user_id"]
    result = clear_user_history(user_id)
    
    return ApiResponse(
        success=result["success"],
        message=result["message"],
        data={"user_id": user_id} if result["success"] else None,
        error=None if result["success"] else result["message"]
    )

@app.get("/mood-tracker", response_model=ApiResponse)
async def mood_tracker(days: int = 7, current_user: dict = Depends(get_current_user)):
    """Get mood tracking data"""
    try:
        user_id = current_user["user_id"]
        mood_history = get_mood_history(user_id, min(days, 30))  # Limit to 30 days
        today_mood = get_daily_emotions(user_id)
        
        return ApiResponse(
            success=True,
            message="Mood data retrieved",
            data={
                "mood_history": mood_history,
                "today": today_mood,
                "days_requested": days
            }
        )
    except Exception as e:
        logger.error(f"Mood tracker error: {e}")
        return ApiResponse(
            success=False,
            message="Failed to get mood data",
            error="Unable to retrieve mood tracking data"
        )

@app.get("/mood-tracker/today", response_model=ApiResponse)
async def today_mood(current_user: dict = Depends(get_current_user)):
    """Get today's mood data"""
    try:
        user_id = current_user["user_id"]
        today_data = get_daily_emotions(user_id)
        
        return ApiResponse(
            success=True,
            message="Today's mood retrieved",
            data=today_data
        )
    except Exception as e:
        logger.error(f"Today mood error: {e}")
        return ApiResponse(
            success=False,
            message="Failed to get today's mood",
            error="Unable to retrieve today's mood data"
        )

@app.post("/mood-entry", response_model=ApiResponse)
async def add_mood_entry(mood_data: MoodEntry, current_user: dict = Depends(get_current_user)):
    """Add a mood entry"""
    if not supabase:
        return ApiResponse(
            success=False,
            message="Mood entry unavailable",
            error="Database connection not available"
        )
    
    try:
        user_id = current_user["user_id"]
        
        result = supabase.table("character_emotions").insert({
            "user_id": user_id,
            "character_name": mood_data.character_name,
            "emotion": mood_data.emotion,
            "led_color": mood_data.led_color,
            "sound_file": mood_data.sound_file,
            "timestamp": datetime.utcnow().isoformat()
        }).execute()
        
        return ApiResponse(
            success=True,
            message="Mood entry added successfully",
            data={
                "mood_entry": {
                    "emotion": mood_data.emotion,
                    "character_name": mood_data.character_name,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )
    except Exception as e:
        logger.error(f"Mood entry error: {e}")
        return ApiResponse(
            success=False,
            message="Failed to add mood entry",
            error="Unable to save mood entry"
        )

@app.get("/test")
async def test_page():
    """Serve the test HTML page"""
    try:
        with open("test_api.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Test page not found")

@app.post("/test-chat")
async def test_chat_endpoint(req: ChatRequest):
    """Test chat endpoint without authentication"""
    return ApiResponse(
        success=True,
        message="Test chat endpoint working",
        data={
            "message": req.message,
            "test": "This endpoint works without auth"
        }
    )

@app.get("/profile", response_model=ApiResponse)
async def get_profile(current_user: dict = Depends(get_current_user)):
    """Get user profile and statistics"""
    if not supabase:
        return ApiResponse(
            success=False,
            message="Profile unavailable",
            error="Database connection not available"
        )
    
    try:
        user_id = current_user["user_id"]
        
        # Get user info
        user_result = supabase.table("users").select("*").eq("id", user_id).execute()
        if not user_result.data:
            return ApiResponse(
                success=False,
                message="User not found",
                error="User profile not found"
            )
        
        user = user_result.data[0]
        
        # Get stats
        try:
            messages_result = supabase.table("emotion_logs").select("id", count="exact").eq("user_id", user_id).execute()
            total_messages = messages_result.count or 0
        except:
            total_messages = 0
        
        mood_summary = get_mood_history(user_id, 7)
        
        return ApiResponse(
            success=True,
            message="Profile retrieved successfully",
            data={
                "user": {
                    "id": user["id"],
                    "username": user["username"],
                    "email": user["email"],
                    "created_at": user["created_at"]
                },
                "stats": {
                    "total_messages": total_messages,
                    "mood_summary": mood_summary
                }
            }
        )
    except Exception as e:
        logger.error(f"Profile error: {e}")
        return ApiResponse(
            success=False,
            message="Failed to get profile",
            error="Unable to retrieve user profile"
        )

# -------------------------
# 🚀 PythonAnywhere WSGI Application
# -------------------------

# This is the WSGI callable that PythonAnywhere will use
application = app

if __name__ == "__main__":
    # For local development only
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")