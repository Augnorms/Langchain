from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence

def chaining_inner_workings():
    model = OllamaLLM(model="phi3:mini")

    prompt_template = ChatPromptTemplate.from_messages(
        [
          ("system", "You are a facts expert who knows facts about {animals}"),
          ("human", "Tell me {fact_count} facts.")
        ]
    )

    #create the combined chain using langchain
    format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
    invoke_model = RunnableLambda(lambda x:model.invoke(x.to_messages()))
    parse_output = RunnableLambda(lambda x: x)

    chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

    result = chain.invoke({"animals": "Elephant", "fact_count":1})

    print(result)

    return



