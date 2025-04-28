import faiss
import openai
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Simple FAISS Test
dimension = 1536  # Same as OpenAI embeddings
index = faiss.IndexFlatL2(dimension)

print("✅ FAISS memory created successfully!")

# Simple OpenAI API Test
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # use gpt-3.5-turbo not gpt-4
        messages=[{"role": "user", "content": "Hello, who are you?"}]
    )
    print("✅ OpenAI GPT-3.5 response:", response.choices[0].message.content)
except Exception as e:
    print("❌ OpenAI Test Failed:", e)