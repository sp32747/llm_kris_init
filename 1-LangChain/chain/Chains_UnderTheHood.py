from os import system
from click import prompt
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda, RunnableSequence


llm = openai = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)
"""
prompt_template = ChatPromptTemplate.from_messages([
   ("system", "you are a comedian tell me a joke about {topic}"),
   ("human","tell me {joke_count} number of jokes")
])

#create individula runnables
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model=RunnableLambda(lambda x: llm.invoke(x.to_messages()))
parse_output=RunnableLambda(lambda x: x.content)

chain=RunnableSequence(
   first=format_prompt,
   middle=[invoke_model],
   last=parse_output
)

response=chain.invoke({"topic":"lawyer", "joke_count":3}) 

print(response) """


# The whole above thing can be run using LCEL chain
prompt_template2 = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a comedian tell me a joke about {topic}"),
        ("human", "tell me {joke_count} number of jokes"),
    ]
)

# chains can be extendedd by adding more runnables, lets uppercase teh output
touppercase = RunnableLambda(lambda x: x.upper())

chain2 = prompt_template2 | llm | StrOutputParser() | touppercase

response2 = chain2.invoke({"topic": "lawyer", "joke_count": 3})
print(response2)
