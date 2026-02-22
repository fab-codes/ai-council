from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()


class AppConfig:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "mistral-nemo")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma")

    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
