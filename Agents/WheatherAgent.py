from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import create_react_agent, AgentExecutor
from langsmith import Client
import requests
import os

load_dotenv()

search_tool = DuckDuckGoSearchRun()

@tool 
def get_WeatherData(city: str) -> str:
    """
    This tool fetches the current weather data for a given city
    """
    url = f'https://api.weatherstack.com/current?access_key={os.getenv("WEATHER_STACK_API_KEY")}&query={city}'
    response = requests.get(url)
    return response.json()

llm = ChatGroq(model="qwen/qwen3.8-27b")
prompt = Client().pull_prompt("hwchase17/react", dangerously_pull_public_prompt=True)

agent = create_react_agent(
    llm = llm,
    tools = [search_tool, get_WeatherData],
    prompt = prompt
)

agent_Executor = AgentExecutor(
    agent = agent, 
    tools = [search_tool, get_WeatherData],
    verbose = True
)

response = agent_Executor.invoke({"input": "Tell me the current weather of the capital of India"})
print(response['output'])
