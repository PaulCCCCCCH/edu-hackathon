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

BASE_URL = "https://api.gmicloud.ai"


def call_text_to_text_api(prompt: str) -> Dict[str, Any]:
    """
    Call GMI Cloud's text-to-text API to generate text based on a prompt.
    
    Args:
        prompt (str): The input prompt for text generation
        
    Returns:
        Dict[str, Any]: API response containing generated text
    """
    url = f"{BASE_URL}/v1/text-to-text"
    headers = {
        "Authorization": f"Bearer {GMI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "prompt": prompt,
        "max_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling text-to-text API: {e}")
        raise


def call_text_to_video_api(prompt: str, output_path: str) -> Dict[str, Any]:
    """
    Call GMI Cloud's text-to-video API to generate a video from text.
    
    Args:
        prompt (str): The input text to generate video from
        output_path (str): Path where the generated video will be saved
        
    Returns:
        Dict[str, Any]: API response containing video generation details
    """
    url = f"{BASE_URL}/v1/text-to-video"
    headers = {
        "Authorization": f"Bearer {GMI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "prompt": prompt,
        "output_format": "mp4",
        "duration": 30,  # seconds
        "style": "realistic"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        # Download the video
        video_url = response.json().get("video_url")
        if video_url:
            video_response = requests.get(video_url)
            video_response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(video_response.content)
                
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling text-to-video API: {e}")
        raise


if __name__ == "__main__":
    # Example usage of text-to-text API
    print("\n=== Text-to-Text Example ===")
    text_prompt = "Explain quantum computing in simple terms"
    try:
        text_response = call_text_to_text_api(text_prompt)
        print("Generated text:", text_response.get("text", "No text generated"))
    except Exception as e:
        print(f"Error: {e}")

    # Example usage of text-to-video API
    print("\n=== Text-to-Video Example ===")
    video_prompt = "A beautiful sunset over the mountains"
    output_video_path = "generated_video.mp4"
    try:
        video_response = call_text_to_video_api(video_prompt, output_video_path)
        print(f"Video generated successfully at: {output_video_path}")
    except Exception as e:
        print(f"Error: {e}")