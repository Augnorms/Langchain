from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os


#define a structure message
messages = [
    SystemMessage("You are an adviser or a consultatnt"),
    HumanMessage("Give me some guidelines to learn well")
]

def chat_model_conversation_history():

    llm = AzureChatOpenAI(
        deployment_name="gpt-4o", 
        api_key=os.environ["GITHUBAPI"],
        azure_endpoint="https://models.inference.ai.azure.com",
        api_version="2024-10-21",   
        model="gpt-4o",  
    )

    result = llm.invoke(messages)

    print(result.content)

 