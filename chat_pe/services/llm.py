import os
import json
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

DEFAULT_MODEL = "llama-3.3-70b-instruct"

class LLMService:
    """
    LLM service that uses the global OpenAI client.
    """
    def __init__(self, model: str = None):
        self.model = model or DEFAULT_MODEL

    def chat_completion(self, messages: list, format: str = "text", stop: list = None):
        """
        Calls the OpenAI client.
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "stop": stop
        }
        
        if format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

