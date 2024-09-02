import os
from dotenv import load_dotenv
import dspy
from loguru import logger

load_dotenv()


def initialize_openai_model():
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL")

    if not api_key or not model_name:
        logger.error("OPENAI_API_KEY or OPENAI_MODEL environment variable is not set")
        raise ValueError("Missing OpenAI configuration")

    model = dspy.OpenAI(model=model_name, api_key=api_key, temperature=0.5)
    dspy.settings.configure(lm=model)
    logger.success(f"OpenAI model {model_name} initialized successfully")
    return model
