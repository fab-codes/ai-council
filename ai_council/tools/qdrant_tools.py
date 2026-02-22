from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from ai_council.config import AppConfig
from ai_council.utils.logger import get_logger

logger = get_logger(__name__)


def create_qdrant_memory(agent_name: str) -> tuple[list, QdrantVectorStore]:
    """Creates the 'remember' and 'recall' tools and the vector store for a specific agent."""
    client = QdrantClient(url=AppConfig.QDRANT_URL)

    embeddings = OllamaEmbeddings(
        model=AppConfig.OLLAMA_EMBEDDING_MODEL,
        base_url=AppConfig.OLLAMA_URL,
    )

    collection_name = f"agent_{agent_name.lower()}"

    if not client.collection_exists(collection_name):
        # Compute vector size from the Ollama model
        embedding_size = len(embeddings.embed_query("test"))
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
        )

        logger.info(f"Created Qdrant collection: {collection_name}")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    @tool
    def remember(content: str) -> str:
        """Save an important experience or thought to your long-term memory.

        Args:
            content: The text to save as a memory. Must be a plain string describing
                     what you experienced or learned. Example: remember(content="Ho imparato che il fuoco scalda")
        """
        vector_store.add_texts([content])
        logger.info(f"[{agent_name}] Memory saved: {content[:60]}...")

        return "Memory saved."

    @tool
    def recall(query: str) -> str:
        """Retrieve relevant memories from your long-term memory. Use it before responding.

        Args:
            query: A plain string describing what you want to remember. Example: recall(query="fuoco e calore")
        """
        docs = vector_store.similarity_search(query, k=3)
        if not docs:
            return "No memories found."
        memories = "\n---\n".join(doc.page_content for doc in docs)
        logger.info(f"[{agent_name}] Memories retrieved for: {query[:60]}")

        return memories

    return [remember, recall], vector_store
