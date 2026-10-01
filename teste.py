import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id ="Qwen/Qwen3-4B-Instruct-2507",
    task ="conversational",
    temperature = 0.7,
    max_new_tokens = 512,
    huggingfacehub_api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
)

chat = ChatHuggingFace(llm=llm)

print(chat.invoke("Hello, how are you?").content)