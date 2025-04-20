import os
from langchain.llms import OpenAI
from langchain.agents import Tool, initialize_agent
from langchain.tools.tavily_search import TavilySearchResults
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import TypedDict

from langchain_community.llms.together import Together
from langgraph.graph import StateGraph

load_dotenv()

# Define state format
class ResearchState(TypedDict):
    query: str
    context: str
    final_answer: str


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

assert TAVILY_API_KEY, "TOGETHER_API_KEY not set in .env"
assert TOGETHER_API_KEY, "OPENAI_API_KEY not set in .env"

tavily_tool = TavilySearchResults()
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    temperature=0.3,
    max_tokens=512
)

research_agent = initialize_agent(
    tools=[
        Tool(
            name="Tavily Search",
            func=tavily_tool.run,
            description="Web search and summarization."
        )
    ],
    llm=llm,
    agent_type="zero-shot-react-description",
    verbose=False
)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful research assistant.

Based on the following research content:

{context}

Answer the user query:
{question}
"""
)

answer_chain = LLMChain(llm=llm, prompt=prompt_template)

def build_graph():
    def research_node(state: ResearchState) -> ResearchState:
        query = state["query"]
        context = research_agent.run(query)
        return {"query": query, "context": context}

    def answer_node(state: ResearchState) -> ResearchState:
        context = state["context"]
        query = state["query"]
        final_answer = answer_chain.run({
            "context": context,
            "question": query
        })
        return {"query": query, "context": context, "final_answer": final_answer}

    graph = StateGraph(ResearchState)
    graph.add_node("research", research_node)
    graph.add_node("answer", answer_node)
    graph.set_entry_point("research")
    graph.add_edge("research", "answer")
    graph.set_finish_point("answer")

    return graph.compile()
