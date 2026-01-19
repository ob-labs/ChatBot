import os
import dotenv

dotenv.load_dotenv()
from langchain_openai import OpenAIEmbeddings


def get_embedding():
    return OpenAIEmbeddings(
        api_key=os.environ["API_KEY"],
        model="embedding-2",
        base_url=os.getenv(
            "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
    )
