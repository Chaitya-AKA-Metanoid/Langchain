import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load Environment Variables
load_dotenv()

# 2. Initialize the Model (Gemini 2.5 Flash)
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

# 3. Create the Prompt Template
# This is "Stateless" because we only pass the current 'input'
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and concise AI assistant."),
    ("human", "{user_input}")
])

# 4. Define Output Parser
parser = StrOutputParser()

# 5. Build the Chain (CampusX style LCEL)
chain = prompt | model | parser

# 6. Simple Chat Loop
print("--- Simple Stateless Chatbot (Type 'exit' to stop) ---")

while True:
    user_query = input("You: ")
    
    if user_query.lower() in ['exit', 'quit', 'bye']:
        print("Bot: Goodbye!")
        break
    
    try:
        # Every time we call invoke, it starts fresh
        response = chain.invoke({"user_input": user_query})
        print(f"Bot: {response}\n")
    except Exception as e:
        print(f"Error: {e}")