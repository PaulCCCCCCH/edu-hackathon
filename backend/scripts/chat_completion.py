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


def call_text_to_text_api(prompt: str) -> Dict[str, Any]:
    """
    Call GMI Cloud's text-to-text API to generate text based on a prompt.
    
    Args:
        prompt (str): The input prompt for text generation
        
    Returns:
        Dict[str, Any]: API response containing generated text
    """
    url = f"{BASE_URL}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GMI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "max_tokens": 200,
        "temperature": 0.7,
        "messages": [{
            "role": "user",
            "content": prompt
        }]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling text-to-text API: {e}")
        raise

if __name__ == "__main__":
    # Example usage of text-to-text API
    print("\n=== Text-to-Text Example ===")
    text_prompt = "Explain quantum computing in simple terms"
    try:
        text_response = call_text_to_text_api(text_prompt)
        print("Generated text:", text_response.get("choices", "No text generated"))
    except Exception as e:
        print(f"Error: {e}")
