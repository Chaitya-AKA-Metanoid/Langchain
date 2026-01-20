import os
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='The sentiment of the feedback')

structured_llm = model.with_structured_output(Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback as positive or negative: {feedback}',
    input_variables=['feedback']
)

classifier_chain = prompt1 | structured_llm

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback: {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback: {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', (lambda x: {"feedback": x.sentiment}) | prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative', (lambda x: {"feedback": x.sentiment}) | prompt3 | model | parser),
    RunnableLambda(lambda x: "Sentiment could not be determined")
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback': 'This is a beautiful phone'})
print(result)

chain.get_graph().print_ascii()
