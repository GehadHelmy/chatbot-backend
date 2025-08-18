# wsgi.py - PythonAnywhere WSGI Configuration
import os
import sys

# Add the project directory to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Set environment variables for PythonAnywhere
os.environ.setdefault('PYTHONPATH', project_dir)

# Import the FastAPI app from main.py
from main import app

# Create WSGI application
application = app

# For debugging on PythonAnywhere
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 