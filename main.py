"""
Hardware Base — AI Hardware Engineering Assistant
Single entry point. Run this in PyCharm (right-click main.py → Run).
Opens: http://127.0.0.1:8000
"""
import uvicorn
from app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=["app"],
    )
