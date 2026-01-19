"""
Store useful variables and configuration
"""

from dotenv import load_dotenv

import os

# Load environment variables from .env file if it exists
load_dotenv()

ENV_NAME: str = os.getenv("ENV_NAME", "Acrobot-v1")
