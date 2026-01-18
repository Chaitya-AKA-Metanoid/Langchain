from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

json_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Write down all the key themes discussed in the review in a list"
        },
        "summary": {
            "type": "string",
            "description": "A brief summary of the review"
        },
        "sentiment": {
            "type": "string",
            "enum": ["pos", "neg"],
            "description": "Return sentiment of the review either negative or positive"
        },
        "pros": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Write down all the pros inside a list"
        },
        "cons": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Write down all the cons inside a list"
        },
        "name": {
            "type": ["string", "null"],
            "description": "Write the name of the reviewer"
        }
    },
    "required": ["key_themes", "summary", "sentiment"]
}

structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke(
    """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse!
    ...
    Review by Nitish Singh
    """
)

print(result)
print(result["name"])   
print(result["summary"])
print(result["sentiment"])
