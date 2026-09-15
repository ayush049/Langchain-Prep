from langchain_core.tools import InjectedToolArg, tool
from typing import Annotated
from dotenv import load_dotenv
import requests
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
import json
import os

load_dotenv()

@tool 
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """ 
    This function fetches the currency covnersion factor between a given base currency and a taget currency.
    """
    url = f'https://v6.exchangerate-api.com/v6/{os.getenv("EXCHANGE_RATE_API_KEY")}/pair/{base_currency}/{target_currency}'

    response = requests.get(url)
    return response.json()

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg])-> float:
    """
    Given a currency conversion rate, this function calculates the target currency value from a given base currency value
    """

    return base_currency_value*conversion_rate

llm = ChatGroq(model="qwen/qwen3.8-27b")

llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

messages = [HumanMessage("What is the conversion factor between INR and USD, and based on that convert 8000 INR to USD")]

ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)

conversion_rate = None
while ai_message.tool_calls:
    for tool_call in ai_message.tool_calls:
        if tool_call['name'] == 'get_conversion_factor':
            tool_messages1 = get_conversion_factor.invoke(tool_call)
            conversion_rate = json.loads(tool_messages1.content)["conversion_rate"]
            messages.append(tool_messages1)
        if tool_call['name'] == 'convert':
            tool_call['args']['conversion_rate'] = conversion_rate
            tool_messages2 = convert.invoke(tool_call)
            messages.append(tool_messages2)
    ai_message = llm_with_tools.invoke(messages)
    messages.append(ai_message)

print(ai_message.content)

# chain = (get_conversion_factor | convert).invoke({"base_currency":"USD", "target_currency":"EUR", "base_currency_value":100})
# print(chain)