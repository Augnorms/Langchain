from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain.schema.output_parser import StrOutputParser

def chaining_messages():
    model = OllamaLLM(model="phi3:mini")

    prompt_template = ChatPromptTemplate.from_messages(
        [
          ("system", "You are a facts expert who knows facts about {animals}"),
          ("human", "Tell me {fact_count} facts.")
        ]
    )

    #create the combined chain using langchain
    chain = prompt_template | model | StrOutputParser()

    result = chain.invoke({"animals": "Elephant", "fact_count":1})

    print(result)

    return



