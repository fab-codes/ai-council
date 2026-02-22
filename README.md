# AI Council

Una simulazione di vita artificiale conversazionale basata su agenti AI con memoria persistente. Tre esseri interagiscono tra loro, si evolvono attraverso le esperienze e costruiscono ricordi individuali nel tempo.

## Come funziona

Tre agenti (`Angela, Eva, Kevin`) si passano messaggi a turno in un grafo [LangGraph](https://github.com/langchain-ai/langgraph). Ogni agente ha una memoria privata su [Qdrant](https://qdrant.tech/) e due abilità:

- `recall` — recupera ricordi rilevanti prima di rispondere
- `remember` — salva esperienze importanti nella memoria a lungo termine

La simulazione termina dopo un numero configurabile di tick (`MAX_TICKS` in `router.py`).

## Stack

| Componente | Tecnologia |
|---|---|
| Grafo agenti | LangGraph |
| LLM (locale) | Ollama (`granite4:3b` o compatibile con tool calling) |
| LLM (cloud) | Google Gemini (`gemini-2.5-flash`) |
| Embedding | Ollama (`embeddinggemma`) |
| Memoria vettoriale | Qdrant |

## Requisiti

- [Ollama](https://ollama.com/) in esecuzione con un modello che supporta tool calling
- [Docker](https://www.docker.com/) per Qdrant
- Python 3.11+

## Setup

```bash
# 1. Avvia Qdrant
docker compose up -d

# 2. Crea il file di configurazione
cp .env.example .env
# Modifica .env con i tuoi valori

# 3. Installa le dipendenze
pip install -e .

# 4. Avvia la simulazione
python -m ai_council.main
```

## Configurazione

Copia `.env.example` in `.env` e imposta le variabili:

```env
# LLM: "ollama" (default) oppure "google"
LLM_CHOICE=ollama

# Ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=granite4:3b
OLLAMA_EMBEDDING_MODEL=embeddinggemma

# Google Gemini (solo se LLM_CHOICE=google)
GOOGLE_API_KEY=
GOOGLE_MODEL_ID=gemini-2.5-flash

# Qdrant
QDRANT_URL=http://localhost:6333
```

> Il modello Ollama deve supportare il tool calling strutturato. Modelli consigliati: `llama3.1:8b`, `granite4:3b`, `qwen2.5:7b`.

## Struttura del progetto

```
ai_council/
├── agents/          # PrimitiveAgent — logica LLM + tool calling
├── config/          # AppConfig — variabili d'ambiente
├── graph/
│   ├── nodes/       # Factory dei nodi LangGraph
│   ├── states/      # GeneralState (messages, tick, last_speaker, last_message)
│   ├── compile_graph.py
│   └── router.py    # Logica di routing round-robin + MAX_TICKS
├── tools/           # Tool remember/recall con Qdrant
├── utils/           # Logger
└── main.py
```
