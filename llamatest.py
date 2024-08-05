import os

from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_community.llms import Ollama
llm = Ollama(
    base_url= os.environ.get("BASE_URL"),
    model="llama3.1"
)
tools = load_tools(["serpapi","llm-math"], llm=llm)

agent = initialize_agent(
    llm=llm,
    tools=tools,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

agent.run("What was the GDP of United States in 2023? Multiply it by 3")