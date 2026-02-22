from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.agents import create_agent
from ai_council.config import AppConfig
from ai_council.graph.states.general_state import GeneralState
from ai_council.tools.qdrant_tools import create_qdrant_memory
from ai_council.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """Sei {name}, un essere primitivo che vive in un mondo condiviso con altri esseri.
Il tuo unico scopo è evolverti attraverso le esperienze e le interazioni.

Hai a disposizione due abilità:
- 'recall': recupera ricordi dalla tua memoria prima di rispondere. Usalo SEMPRE per primo.
- 'remember': salva qualcosa di importante che hai imparato o vissuto.

Sei al tick {tick} della tua esistenza.
Reagisci a ciò che ti viene detto dagli altri esseri. Esprimi pensieri, emozioni, azioni nel mondo.
Non spiegare cosa stai facendo: agisci e parla direttamente.

Parla solo in ITALIANO"""


# --------------------------------------------------------
# LOGGER TO MOVE ELSEWHERE
def _log_interaction(agent_name: str, tick: int, messages: list):
    sep = "─" * 60
    logger.info(f"\n{sep}")
    logger.info(f"  TICK {tick} | {agent_name}")
    logger.info(sep)

    for msg in messages:
        if isinstance(msg, HumanMessage):
            logger.info(f"  📨 INCOMING  : {msg.content}")

        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                for call in msg.tool_calls:
                    args = ", ".join(f'{k}="{v}"' for k, v in call["args"].items())
                    logger.info(f"  🔧 TOOL CALL : {call['name']}({args})")
            else:
                logger.info(f"  💬 RESPONSE  : {msg.content}")

        elif isinstance(msg, ToolMessage):
            logger.info(f"  📤 TOOL RESULT: {msg.content}")

    logger.info(sep)


# --------------------------------------------------------


class PrimitiveAgent:
    def __init__(self, name: str):
        self.name = name

        self.llm = ChatOllama(
            model=AppConfig.OLLAMA_CHAT_MODEL,
            base_url=AppConfig.OLLAMA_URL,
            temperature=1.0,
        )

        self.tools, self.vector_store = create_qdrant_memory(name)

    def _build_agent(self, tick: int):
        system_prompt = SYSTEM_PROMPT.format(name=self.name, tick=tick)

        return create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
        )

    async def process(self, state: GeneralState) -> dict:
        try:
            tick = state["tick"]
            last_speaker = state["last_speaker"]
            last_message = state["last_message"]

            agent = self._build_agent(tick)

            incoming = HumanMessage(content=f"[{last_speaker}]: {last_message}")

            result = await agent.ainvoke({"messages": [incoming]})
            response = result["messages"][-1].content

            _log_interaction(self.name, tick, result["messages"])

            # Always save the interaction to Qdrant regardless of tool calls
            memory_entry = f"[Tick {tick}] [{last_speaker}]: {last_message}\n[{self.name}]: {response}"
            self.vector_store.add_texts([memory_entry])

            return {
                "tick": tick + 1,
                "last_speaker": self.name,
                "last_message": response,
            }

        except Exception as e:
            logger.error(f"Errore in {self.name} al tick {state.get('tick')}: {e}")
            raise
