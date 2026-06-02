import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model = os.getenv("OPENAI_MODEL", "gpt-5.5")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in .env")

if not base_url:
    raise ValueError("OPENAI_BASE_URL is not set in .env")

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "用一句话解释什么是神经网络。"
        }
    ],
)

print(response.choices[0].message.content)
