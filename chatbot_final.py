import os
import json
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

class SupportBot:
    def __init__(self, prompt_path="prompt.json"):
        if not os.path.exists(prompt_path) or os.stat(prompt_path).st_size == 0:
            raise FileNotFoundError(f"'{prompt_path}' is missing. I need my instructions to help you.")

        with open(prompt_path, "r") as f:
            self.config = json.load(f)
            
        self.model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        self.system_prompt = SystemMessage(content=self.config["system_message"])
        self.chat_history = []
        self.memory_window = 8 # A bit more memory for deep conversations

    def get_support_response(self, user_text):
        self.chat_history.append(HumanMessage(content=user_text))
        payload = [self.system_prompt] + self.chat_history[-self.memory_window:]
        response = self.model.invoke(payload)
        self.chat_history.append(AIMessage(content=response.content))
        return response.content

try:
    bot = SupportBot()
    os.system('cls' if os.name == 'nt' else 'clear')
    print("================================================")
    print("       VIRTUAL THERAPY & SUPPORT SPACE          ")
    print("================================================")
    print("Therapist: Hello. I am here to listen. How are you feeling today?")
    print("(Type 'exit' to end our session)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\nTherapist: Take care of yourself. I'm here whenever you need to talk.")
            break
        if not user_input:
            continue

        print("Reflecting...", end="\r")
        
        try:
            answer = bot.get_support_response(user_input)
            print(" " * 30, end="\r")
            print(f"Therapist: {answer}\n")
        except Exception as e:
            print(f"\nTherapist: I'm sorry, I'm having a hard time connecting right now. Let's try again.")

except Exception as e:
    print(f"Startup Error: {e}")