from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.llms import Ollama  # Corrected import
from datetime import datetime
from langchain.tools import tool
from langchain import hub
import os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory

load_dotenv()

# For LangSmith usage (if you have an account)
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGSMITH')  # Not endpoint!
os.environ["SERPAPI_API_KEY"] = os.getenv('SERPAPI')

@tool("get_current_time", description="Returns the current date and time.")
def get_current_time(input: str = "") -> str:
    return datetime.now().isoformat()

def agentic_model():
    memory = ConversationBufferMemory()

    # Initialize the language model
    llm = Ollama(model="tinyllama")  # Corrected class

    # Load predefined tools
    tools = load_tools(['serpapi', 'llm-math'], llm=llm)
    tools.append(get_current_time)

    # Retrieve the prompt template
    prompt = hub.pull("hwchase17/structured-chat-agent")

    # Initialize agent components
    agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

    conversation_history = memory.chat_memory.messages

    print(conversation_history)
    
    return agent_executor.invoke({"input": "what is ollama"})
