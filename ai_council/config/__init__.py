from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()


class AppConfig:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma")

    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_MODEL_ID = os.getenv("GOOGLE_MODEL_ID", "gemini-2.5-flash")

    LLM_CHOICE = os.getenv("LLM_CHOICE", "ollama")
