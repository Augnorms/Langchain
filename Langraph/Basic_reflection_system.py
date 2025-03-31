from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureOpenAI
from typing import List, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph 
from langchain_core.output_parsers import StrOutputParser
import os

# Initialize the Azure OpenAI LLM
llm = AzureOpenAI(
        deployment_name="gpt-4o", 
        api_key=os.environ["GITHUBAPI"],
        azure_endpoint="https://models.inference.ai.azure.com",
        api_version="2023-05-15"
    )

# Modify prompts to be more safety-conscious
generate_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            "You are a tech Twitter assistant. Create engaging, positive tweets about AI trends."
            "Keep posts under 280 characters. Avoid controversial topics."
            "If given feedback, revise accordingly while maintaining positivity."
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            "You are a social media analyst. Provide constructive feedback on tweets."
            "Focus on engagement, clarity, and positivity. Suggest improvements for style and virality."
            "Always maintain a professional tone."
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)
#create a chain
generate_chain = generate_prompt | llm | StrOutputParser()
reflection_chain = reflection_prompt | llm | StrOutputParser()

def generate_node(state):
    return generate_chain.invoke({
        "messages": state
    })

def reflect_node(state):
    response =  reflection_chain.invoke({
        "messages": state
    })

    return [HumanMessage(content=response)]



#main 1function execution
def basic_reflection():
    graph = MessageGraph()

    REFLECT = "reflect"
    GENERATE = "generate"

    graph.add_node(GENERATE, generate_node)
    graph.add_node(REFLECT, reflect_node)

    graph.set_entry_point(GENERATE)

    def should_continue(state):
        # Track iterations instead of message count
        if len(state) >= 4:  # Allow 2 iterations (generate -> reflect -> generate)
            return END
        return REFLECT
    
    graph.add_conditional_edges(GENERATE, should_continue)
    graph.add_edge(REFLECT, GENERATE)

    #use to complie the graph
    app = graph.compile()
    
    #use to draw the flow diagram
    print(app.get_graph().draw_mermaid())
    app.get_graph().print_ascii()

    response = app.invoke(HumanMessage(content="AI Agent taking over content creation"))

    print(response)
