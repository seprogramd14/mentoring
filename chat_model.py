from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

class ChatBot:
    def __init__(self, description):
        self.prompt = PromptTemplate.from_template(template=description)
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        self.parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.parser

    def __call__(self, information):
        return self.chain.invoke(input={"information": information})