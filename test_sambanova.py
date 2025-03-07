import os
from langchain_sambanova import ChatSambaNovaCloud
from dotenv import load_dotenv

# Load API key
load_dotenv()
API_KEY = os.getenv("SAMBANOVA_API_KEY")

# Initialize SambaNova model
llm = ChatSambaNovaCloud(model="Qwen2.5-Coder-32B-Instruct", api_key=API_KEY)

# Test prompt
prompt = (
    "Generate a Python function that calculates the factorial of a number."
)

# Generate response
response = llm.invoke(prompt)

print("SambaNova Response:\n", response)
