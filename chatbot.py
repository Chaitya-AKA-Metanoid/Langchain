import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# Initialize Gemini 2.5 Flash
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# Initial system context
chat_history = [
    SystemMessage(content='You are a helpful AI assistant.')
]

print("--- Gemini 2.5 Chat Session ---")

while True:
    # Use .strip() to remove accidental spaces/newlines
    raw_input = input('You: ').strip()
    
    # SOLUTION 1: Defensive check for empty input
    if not raw_input:
        print("Assistant: (Ignoring empty message...)")
        continue 
    
    if raw_input.lower() == 'exit':
        break
    
    # Add valid input to history
    chat_history.append(HumanMessage(content=raw_input))
    
    try:
        # Gemini requires a HumanMessage in the list to work.
        # Since we just appended one, this will now succeed.
        result = model.invoke(chat_history)
        
        chat_history.append(AIMessage(content=result.content))
        print("AI:", result.content)
        
    except Exception as e:
        # Catch-all for API errors (like quota or safety blocks)
        print(f"API Error: {e}")
        # Remove the last message if the call failed so history stays clean
        chat_history.pop()