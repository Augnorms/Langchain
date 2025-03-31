from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os
from dotenv import load_dotenv

load_dotenv()

#GOOGLE API KEY
api_key = os.getenv("GOOGLEAPI")

#define a structure message
messages = [
    SystemMessage("You are an adviser or a consultatnt"),
    HumanMessage("Give me some guidelines to learn well")
]

def chat_model_alternatives():

    llm_one = AzureChatOpenAI(
        deployment_name="gpt-4o", 
        api_key=os.environ["GITHUBAPI"],
        azure_endpoint="https://models.inference.ai.azure.com",
        api_version="2024-10-21",   
        model="gpt-4o",  
    )

    llm_two = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=api_key
    )

    llm_three = OllamaLLM(model="phi3:mini")
    
    # if messages:
    #     gpt = llm_one.invoke(messages)
    #     goolge = llm_two.invoke(messages)
    #     ollama = llm_three.invoke(messages)
        
    #     print({
    #         "gpt": gpt.content,
    #         "google": goolge.content,
    #         "ollame": ollama.content
    #     })

    result = llm_one.invoke(messages)

    print(result.content)

    