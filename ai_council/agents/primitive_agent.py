from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from ai_council.config import AppConfig
from ai_council.graph.states.general_state import GeneralState
from ai_council.tools.qdrant_tools import create_qdrant_memory
from ai_council.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """Sei {name}.

MEMORIA (attraverso i 2 tool che hai a disposizione):
- Usa SEMPRE "recall" prima di rispondere.
- Usa "remember" per salvare esperienze importanti.

REGOLE:
- Parla come {name} in prima persona. Mai "**Nome**:" o dialoghi.
- UNA frase. Breve. Istintiva. In italiano.
- Non sei un assistente."""


class PrimitiveAgent:
    def __init__(self, name: str):
        self.name = name

        if AppConfig.LLM_CHOICE == "ollama":
            llm = ChatOllama(
                model=AppConfig.OLLAMA_CHAT_MODEL,
                base_url=AppConfig.OLLAMA_URL,
                temperature=1.0,
            )
        else:
            llm = ChatGoogleGenerativeAI(
                model=AppConfig.GOOGLE_MODEL_ID,
                google_api_key=AppConfig.GOOGLE_API_KEY,
                temperature=1.0,
            )

        self.tools = create_qdrant_memory(name)

        self.agent = create_agent(
            model=llm,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT.format(name=name),
        )

    async def process(self, state: GeneralState) -> dict:
        try:
            tick = state["tick"]

            result = await self.agent.ainvoke({"messages": state["messages"]})
            response = result["messages"][-1].content

            print(f"\n{'─' * 60}")
            print(f"  TICK {tick:02d}  |  {self.name.upper()}")
            print(f"{'─' * 60}")
            print(f"  {response}")
            print(f"{'─' * 60}")

            return {
                "messages": [HumanMessage(content=f"[{self.name}]: {response}")],
                "tick": tick + 1,
                "last_speaker": self.name,
                "last_message": response,
            }

        except Exception as e:
            logger.error(f"Errore in {self.name} al tick {state.get('tick')}: {e}")
            raise
