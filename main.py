import streamlit as st
from langchain_resturant_file import generate_restaurant_name_and_item
from langchain_agent_file import agentic_model
from langchain_vector_embedding_project import chunk_summarizer

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

st.title("New Research Tool")
main_placeholder = st.empty()

user_query = st.text_input("Enter your query here:")


st.sidebar.title("News Article URLs")

urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_clicked = st.sidebar.button("Process URLs")

if process_clicked:
    if urls and user_query:
        main_placeholder.text("Loading data...")
        result = chunk_summarizer(urls, user_query)
        main_placeholder.text("Done.")

        st.subheader("your answer is: ")
        st.write(result["answer"])
        st.write(result.get("source"))
    else:
        st.sidebar.error("Please provide at least one URL and enter a query.")