import os
import requests
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()

# GMI Cloud API configuration
GMI_API_KEY = os.getenv("GMI_API_KEY")
if not GMI_API_KEY:
    raise ValueError("GMI_API_KEY environment variable is not set")

BASE_URL = "https://api.gmi-serving.com"


def list_video_models() -> Dict[str, Any]:
    """
    List available video models from GMI Cloud.
    
    Returns:
        Dict[str, Any]: API response containing available video models
    """
    url = f"{BASE_URL}/video/models"
    headers = {
        "Authorization": f"Bearer {GMI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error listing video models: {e}")
        raise

def list_models() -> Dict[str, Any]:

    url = f"{BASE_URL}/v1/models"
    headers = {
        "Authorization": f"Bearer {GMI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error listing video models: {e}")
        raise


def main():
    """Main function to list video models and print the results."""
    try:
        # models = list_video_models()
        models = list_models()
        print("\n=== Available Video Models ===")
        for model in models.get("data", []):
            print(f"- ID: {model.get('id', 'Unknown')}")
            print(f"  object: {model.get('object', 'Unknown')}")
            print(f"  Owned By: {model.get('owned_by', 'No description')}")
            print("")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
