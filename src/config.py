"""
Store useful variables and configuration
"""

from dotenv import load_dotenv

import os

# Load environment variables from .env file if it exists
load_dotenv()

ENV_NAME = os.getenv("ENV_NAME", "MountainCarContinuous-v0")
