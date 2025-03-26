from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel, RunnableBranch


def chain_real_wolrd_considtional():
    model = OllamaLLM(model="phi3:latest")

    positive_feedback_template = ChatPromptTemplate(
        [
           ("system", "You are a helpful assistant."),
           ("human", "Generate a thank you note for this positive feedback: {feedback}")
        ]
    )

    negative_feedback_template = ChatPromptTemplate(
        [
           ("system", "You are a helpful assistant."),
           ("human", "Generate a response addressing this negative feedback: {feedback}")
        ]
    )

    neutral_feedback_template = ChatPromptTemplate(
        [
           ("system", "You are a helpful assistant."),
           ("human", "Generate a response for more detail for this neutral feedback: {feedback}")
        ]
    )

    escalate_feedback_template = ChatPromptTemplate(
        [
           ("system", "You are a helpful assistant."),
           ("human", "Generate a message to escalate this feedback to a human agent: {feedback}")
        ]
    )

    classification_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "Classify the sentiment of this feedback as positive, negative, nuetral or escalate: {feedback}")
        ]
    )

    #define branches
    branches = RunnableBranch(
        (
            lambda x: "positive" in x,
            positive_feedback_template | model | StrOutputParser()
        ),
        (
            lambda x: "negative" in x,
            negative_feedback_template | model | StrOutputParser()
        ),
        (
            lambda x: "neutral" in x,
            neutral_feedback_template | model | StrOutputParser()
        ),
        escalate_feedback_template | model | StrOutputParser()
    )

    #classification chain
    classification_chain = classification_template | model | StrOutputParser()

    chain = classification_chain | branches

    review = "The product is terrible. It broke after one use and the quality is very poor"

    result = chain.invoke({"feedback": review})

    print(result)

    return