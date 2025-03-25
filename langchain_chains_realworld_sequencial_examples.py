from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda

def chaining_real_world():
    model = OllamaLLM(model="phi3:latest")

    animal_template = ChatPromptTemplate.from_messages(
        [
          ("system", "You are a facts expert who knows facts about {animals}"),
          ("human", "Tell me {fact_count} facts.")
        ]
    )

    translation_template = ChatPromptTemplate.from_messages(
        [
        ("system", "You are a professional translator. Your job is to translate the given text into French and return only the translated text."),
        ("human", "Please translate this text into French:\n{text}")
        ]
    )


    prepare_to_translate = RunnableLambda(lambda output: {"text": output, "language":"french"} )

    #create the combined chain using langchain
    chain = animal_template | model | StrOutputParser() | prepare_to_translate | translation_template | model | StrOutputParser()
    
    result = chain.invoke({"animals": "cat", "fact_count":1})

    print(result)

    return



