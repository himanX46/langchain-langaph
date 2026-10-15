import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
#from ollama import Tool
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
#from tavily import TavilyClient
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field

class Source(BaseModel):
    """Scheme for a source used by an agent"""
    url:str = Field(description="The URL of the source") 

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List pf sources unsed to generate the answer")

load_dotenv()
#tavily = TavilyClient()

#@tool
def search(query:str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search results
    """
    print(f"Searching for {query}")
    #return tavily.search(query=query)

#llm = ChatOllama(model="gemma3:270m", temperature=0)
llm = ChatOllama(model="qwen3:4b", temperature=0)
#tools = [search]
tools = [TavilySearch()]
#agent = create_agent(model=llm, tools=tools)
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main() -> None:
    
    print("Hello from langchain-course!")
    #information = """
    #Hanuman (/ˈhʌnʊˌmɑːn/; Sanskrit: हनुमान्, IAST: Hanumān),[5] also known as Maruti, Bajrangabali, and Anjaneya,[6] is a deity in Hinduism, revered as a divine vanara, and a devoted companion of Lord Rama. Central to the Ramayana, Hanuman is celebrated for his unwavering devotion to Rama and is considered a chiranjivi. He is traditionally believed to be the spiritual offspring of the celestial wind-god Vayu.[7][8] In Shaiva tradition, he is regarded as an incarnation of Shiva, while in most of the Vaishnava traditions he is the son and incarnation of Vayu. His tales are recounted not only in the Ramayana but also in the Mahabharata and various Puranas. Devotional practices centered on Hanuman were not prominent in these texts or in early archaeological evidence. His theological significance and the cultivation of a devoted following emerged roughly a millennium after the Ramayana was composed, during the second millennium CE.[9]
    #"""

    #summary_template = """
    #Given the information{information} about a person I want you to create:
    #1. A short summary
    #2. Two interesting facts about them
    #"""
    #summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)
    #llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    #llm = ChatOllama(model="gemma3:270m", temperature=0)
    #chain = summary_prompt_template | llm
    #response = chain.invoke({"information": information})
    #print(response.content)
    #agent = create_agent(llm=llm, tools=[search], verbose=True)
    #response = agent.invoke({"input": "What is the capital of France?"})
    #print(response.content)

    ##########################
    
    response = agent.invoke({"messages": [HumanMessage(content="Search for top 3 job openings in Agentic AI space in India from LinkedIn")]})
    for m in response["messages"]:
        m.pretty_print()
    print(response.get("structured_response"))

    ##########################

if __name__ == "__main__":
    main()
