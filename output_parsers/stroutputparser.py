from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Initialize Gemini 2.5 Flash
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. \n {text}',
    input_variables=['text']
)

# Step 1: Generate Report
prompt1 = template1.invoke({'topic': 'black hole'})
result = model.invoke(prompt1)

# Step 2: Manually extract .content and pass to Summary
prompt2 = template2.invoke({'text': result.content})
result1 = model.invoke(prompt2)

print(result1.content)
