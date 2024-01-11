import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate 
from langchain.chains import LLMChain


davinci = OpenAI(model_name='gpt-3.5-turbo-instruct')

# build prompt template for simple question-answering
template = """Question: {question}

Answer: """
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(
    prompt=prompt,
    llm=davinci
)

question = "Which NFL team won the Super Bowl in the 2010 season?"

print(llm_chain.invoke(question))