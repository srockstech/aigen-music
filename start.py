import os
import uvicorn

if __name__ == "__main__":
    # Get port with fallback to 8000
    try:
        port = int(os.environ.get("PORT", "8000"))
    except ValueError:
        port = 8000
    
    print(f"Starting server on port {port}")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    ) 