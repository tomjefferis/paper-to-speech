import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Google API configuration
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

def get_google_api_key():
    """
    Retrieve the Google API key from environment variables.
    
    Returns:
        str: The Google API key or None if not found
    
    Raises:
        ValueError: If the API key is not set
    """
    api_key = GOOGLE_API_KEY
    if not api_key:
        raise ValueError("Google API key not found. Please set GOOGLE_API_KEY in your .env file.")
    return api_key
