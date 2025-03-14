from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLEAPI")
db_user = os.getenv("DBUSER")
db_host = os.getenv("DBHOST")
db_password = os.getenv("DBPASSWORD")
db_name = os.getenv("DBNAME")

# Custom prompt template
TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query. Execute the query and return the answer based on the result. 

Follow this format:
Question: {input}
SQLQuery: SQL query to run
SQLResult: Result from the database
Answer: Extract the value from SQLResult. Example: If SQLResult is [(Decimal('15'),)], 
Answer: 15. Return "None" if no data.

Use the following tables:
{table_info}

Question: {input}

"""

CUSTOM_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect"], template=TEMPLATE
)


def querydatabase_ai():
    # Initialize database connection
    db = SQLDatabase.from_uri(
        f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}", 
        sample_rows_in_table_info=3,
    )

    # Initialize the LLM (Google Gemini)
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-1.5-flash",
    #     temperature=0,
    #     google_api_key=api_key
    # )

    llm = OllamaLLM(model="phi3:mini") 

    # Create SQLDatabaseChain
    db_chain = SQLDatabaseChain.from_llm(llm, db, prompt=CUSTOM_PROMPT, verbose=False)

    # Execute query
    # query = "which of the brands has the most stock_quantity?"

    input_values = {
        "query": "which of the brands has the most stock_quantity?",  
        "table_info": db.get_table_info(),
        "dialect": db.dialect,
    }

    try:
        response = dict(db_chain.invoke(input_values))
    except Exception as e:
        # Catch any SQL or chain errors and return a fallback answer
        response = {"error": f"SQL produced an error: {str(e)}"}
    
    print(f"Final Answer: {response['result']}")


