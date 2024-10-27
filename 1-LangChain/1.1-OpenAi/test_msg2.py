from os import system
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from openai import chat


llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

chat_history = []

system_msg = SystemMessage(content="you are an amazing assistant")

chat_history.append(system_msg)

while True:
    qry = input("You : ")
    if qry.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=qry))
    res = llm.invoke(chat_history)
    response = res.content
    chat_history.append(AIMessage(content=response))
    print(f"AI: {response}")

print("-=======chat_history==============-")

print(chat_history)
