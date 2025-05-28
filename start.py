import os
from api import app

if __name__ == "__main__":
    # Get port with fallback to 8000
    try:
        port = int(os.environ.get("PORT", "8000"))
    except ValueError:
        port = 8000
    
    print(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port) 