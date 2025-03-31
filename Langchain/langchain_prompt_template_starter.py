from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# def prompt_template_starter():
#     llm = OllamaLLM(model="phi3:mini")

#     template = """ 
#         write a {tone} email to {company} expressing interest in {position}, 
#         mentioning {skill} as a key strength. Keep it to 4 lines max    
#     """

#     prompt_template = ChatPromptTemplate.from_template(template)

#     prompt = prompt_template.invoke({
#         "tone": "energetic",
#         "company": "samsung",
#         "position": "AI Engineer",
#         "skill": "AI"
#     })

#     result = llm.invoke(prompt)

#     print(result)
#     return

def prompt_template_starter():
    llm = OllamaLLM(model="phi3:mini")

    message = [
        ("system", "You are a comedian who tells jokes about {topic}"),
        ("human", "Tell me {joke_count} jokes.")
    ]

    prompt_template = ChatPromptTemplate.from_messages(message)

    prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})

    result = llm.invoke(prompt)

    print(result)

    return
