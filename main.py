import streamlit as st
from langchain_resturant_file import generate_restaurant_name_and_item
from langchain_agent_file import agentic_model
from langchain_vector_embedding_project import chunk_summarizer
from Langchain_dbquery_project import querydatabase_ai
from langchain_chat_model_starter import chat_model_starter
from Langchain_model_conversational_history import chat_model_conversation_history
from Langchain_model_alternatives import chat_model_alternatives
from langchain_conversational_with_chat_history import remembering_chat_conversation
from langchain_prompt_template_starter import prompt_template_starter
from langchain_chains import chaining_messages
from langchain_chain_inner_workings import chaining_inner_workings
from langchain_chains_realworld_sequencial_examples import chaining_real_world_sequencial
from langchain_chains_realworld_parallel_example import chain_real_wolrd_parallel
from langchain_chains_relaworld_conditional_example import chain_real_wolrd_considtional
from langchain_Retrieval_augmented_gen import retrieval_augmented_gen
from langchain_agent_101 import chain_agent

#resturant code

# st.title("Restaurant Name Generator")
# cuisine = st.sidebar.selectbox("Pick a cuisine", ('', 'Indian', 'Italian', 'Ghanaian', 'Mexican'))

# if cuisine:
#     response = generate_restaurant_name_and_item(cuisine)
    
#     st.header(response['restaurant_name'])  # Display the restaurant name
    
#     st.write('**Menu Items**')
#     for item in response['menu_items']:
#         st.write('-', item.strip())  # Display each menu item


#vector embedding project

# st.title("New Research Tool")
# main_placeholder = st.empty()

# user_query = st.text_input("Enter your query here:")


# st.sidebar.title("News Article URLs")

# urls = []

# for i in range(3):
#     url = st.sidebar.text_input(f"URL {i+1}")
#     if url:
#         urls.append(url)

# process_clicked = st.sidebar.button("Process URLs")

# if process_clicked:
#     if urls and user_query:
#         main_placeholder.text("Loading data...")
#         result = chunk_summarizer(urls, user_query)
#         main_placeholder.text("Done.")

#         st.subheader("your answer is: ")
#         st.write(result["answer"])
#         st.write(result.get("source"))
#     else:
#         st.sidebar.error("Please provide at least one URL and enter a query.")

# querydatabase_ai()

# chat_model_starter()

# chat_model_conversation_history()

# chat_model_alternatives()
# remembering_chat_conversation()
# prompt_template_starter()
# chaining_messages()
# chaining_inner_workings()
# chaining_real_world_sequencial()
# chain_real_wolrd_parallel()
# chain_real_wolrd_considtional()
# retrieval_augmented_gen()
chain_agent()