import datetime
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
import os
load_dotenv()

@tool("get_current_time", description="Returns the current date and time.")
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    current_time = datetime.datetime.now()
    return current_time.strftime(format)

def chain_agent():
    model = OllamaLLM(model="phi3:latest")
    llm = AzureChatOpenAI(
        deployment_name="gpt-4o", 
        api_key=os.environ["GITHUBAPI"],
        azure_endpoint="https://models.inference.ai.azure.com",
        api_version="2024-10-21",   
        model="gpt-4o",  
    )

    query = "What is the current time in USA (you are in Ghana)? just show the current time and not the date"
    # prompt_template = PromptTemplate.from_template("{input}")
    prompt_template = hub.pull("hwchase17/react")

    tools = [get_system_time]

    # chain = prompt_template | model | StrOutputParser()
    agent = create_react_agent(llm, tools, prompt_template)

    # result = chain.invoke({"input": query})
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    agent_executor.invoke({"input": query})

    # print(result)

    return