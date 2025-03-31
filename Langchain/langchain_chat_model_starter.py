from langchain_openai import AzureChatOpenAI
import os


def chat_model_starter():

    llm = AzureChatOpenAI(
        deployment_name="gpt-4o", 
        api_key=os.environ["GITHUBAPI"],
        azure_endpoint="https://models.inference.ai.azure.com",
        api_version="2024-10-21",   
        model="gpt-4o",  
    )

    result = llm.invoke("what is the language for France")

    print(result.content)

    return 