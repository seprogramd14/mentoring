from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

description = "이 {information} 에 대해서 "
information = "총"

prompt = PromptTemplate.from_template(template=description)
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

res = chain.invoke(input={"information": information})
print(res)