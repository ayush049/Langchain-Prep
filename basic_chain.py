from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

model = ChatGroq(model="qwen/qwen3.6-27b", max_tokens=250)
parser = StrOutputParser()
chain = prompt | model | parser
result = chain.invoke({'topic': 'langchain'})
print(result)

try:
    chain.get_graph().print_ascii()
except Exception as e:
    print(f"\n(Note: Graph printing requires grandalf package: {e})")