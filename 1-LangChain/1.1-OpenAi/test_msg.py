from os import system
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage


llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)


messages = [
    SystemMessage(content="You are a python expert."),
    HumanMessage(content="what is 2+2"),
    AIMessage(content="4"),
    HumanMessage(content="write code for adding 2 numbers"),
]

res = llm.invoke(messages)
print(res.content)
