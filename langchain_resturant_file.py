from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate


def generate_restaurant_name_and_item(cuisine):
    llm = OllamaLLM(model="tinyllama")  

    # Step 1: Generate a restaurant name
    prompt_template_name = PromptTemplate(
        input_variables=["cuisine"], 
        template="Suggest a single fancy restaurant name for {cuisine} cuisine. Only return the name, nothing else."
    )

    # Step 2: Generate menu items for that restaurant
    prompt_template_items = PromptTemplate(
        input_variables=["restaurant_name"],  
        template="Suggest 5 menu items for {restaurant_name}. Return them as a comma-separated list."
    )

    # Generate the restaurant name
    restaurant_name = llm.invoke(prompt_template_name.format(cuisine=cuisine)).strip()

    # Generate menu items
    menu_items = llm.invoke(prompt_template_items.format(restaurant_name=restaurant_name)).strip()


    # Return as a structured dictionary
    return {
        "restaurant_name": restaurant_name,
        "menu_items": menu_items.split(",")  # Convert string to list
    }
