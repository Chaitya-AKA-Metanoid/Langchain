import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Explicitly check if the key is loaded
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found. Check your .env file.")
else:
    # Use "models/gemini-1.5-flash" to avoid the 404 error
    model = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
    temperature=1.0
    )

    try:
        print("Connecting to Gemini...")
        result = model.invoke("Write a 5 line poem on cricket")
        print("\n--- Response Received ---")
        print(result.content)
    except Exception as e:
        print(f"\nFailed: {e}")