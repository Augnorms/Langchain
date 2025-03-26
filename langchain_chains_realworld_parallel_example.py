from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel


def analyze_plot(plot):
    plot_template = ChatPromptTemplate(
        [
            ("system", "You are a movie critic."),
            ("human", "Analyze the plot: {plot}. What are its strenght and weekness")
        ]
    )
    return plot_template.format_prompt(plot=plot)

def analyse_character(character):
    character_template = ChatPromptTemplate(
        [
            ("system", "You are a movie critic."),
            ("human", "Analyze the character: {character}. What are its strenght and weekness")
        ]
    )
    return character_template.format_prompt(character=character)

def combine_parallel_analyser(plot_analyses, character_analysis):
    return f"Plot Analyses:\n{plot_analyses}\n\nCharacter Analyses:\n{character_analysis}"


def chain_real_wolrd_parallel():
    model = OllamaLLM(model="phi3:latest")

    summary_template = ChatPromptTemplate(
        [
            ("system", "You are a movie critic."),
            ("human", "Provide a brief summary of the movie {movie_name}")
        ]
    )

    plot_branch_chain = (
        RunnableLambda(lambda x:analyze_plot(x)) | model | StrOutputParser()
    )

    character_branch_chain = (
        RunnableLambda(lambda x:analyse_character(x)) | model | StrOutputParser()
    )

    chain = (
        summary_template 
        | model
        | StrOutputParser()
        | RunnableParallel(branches={"plot": plot_branch_chain, "character":character_branch_chain})
        | RunnableLambda(lambda x:combine_parallel_analyser(x["branches"]["plot"], x["branches"]["character"]))
    )

    result = chain.invoke({"movie_name": "Inception"})

    print(result)

    return