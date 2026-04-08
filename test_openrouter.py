from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

for model_name in ["text-embedding-3-small", "openai/text-embedding-3-small", "openai/text-embedding-3-small:beta", "text-embedding-3-small:beta"]:
    print(f"\n--- Testing {model_name} ---")
    embed = OpenAIEmbeddings(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        model=model_name
    )
    try:
        vec = embed.embed_query("test")
        print(f"Success! Length: {len(vec)}")
    except Exception as e:
        print(f"Failed: {e}")
