import os
from dotenv import load_dotenv
from together import Together

# ---------------------------------------------------------------------
# 1. Load API Key from .env
# ---------------------------------------------------------------------
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY is not set. Please define it in your .env file.")

# ---------------------------------------------------------------------
# 2. Initialize Together Client
# ---------------------------------------------------------------------
client = Together(api_key=TOGETHER_API_KEY)

# ---------------------------------------------------------------------
# 3. Make Chat Completion Request
# ---------------------------------------------------------------------
try:
    stream = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[{"role": "user", "content": "What are the top 3 things to do in New York?"}],
        stream=True,
    )

    # Print the response content
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
except Exception as e:
    print(f"An error occurred: {e}")
