from langchain_ollama import OllamaLLM
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os
from db_connect import save_message_to_db

def remembering_chat_conversation():
    llm = OllamaLLM(model="phi3:mini")

    # Save the initial system message directly to the database.
    system_message = SystemMessage(content="You are a helpful AI assistant.")
    save_message_to_db("System", system_message.content)

    # Chat loop
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        # Save user message to the database
        human_message = HumanMessage(content=query)
        save_message_to_db("User", human_message.content)

        # Get AI response (here we pass the current query; adjust as needed)
        result = llm.invoke(query)
        response = result.content if hasattr(result, 'content') else result
        print(f"AI: {response}")

        # Save AI message to the database
        save_message_to_db("AI", response)
  
        