import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_DEPLOYMENT = os.getenv("MODEL_DEPLOYMENT", "gpt-4.1")
GRADER_MODEL_DEPLOYMENT = os.getenv("GRADER_MODEL_DEPLOYMENT", "gpt-5.4")
