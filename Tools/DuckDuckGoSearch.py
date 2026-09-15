from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()
results = search_tool.invoke('Top news today')
print(results)
