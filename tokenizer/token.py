from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

# Automatically uses GEMINI_API_KEY from environment
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

response = client.models.generate_content(
    model="gemini-2.5-flash",   # fast & free tier model
    contents="Explain AI in simple words"
)

print(response.text)