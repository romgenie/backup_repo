import os
import requests
import json
from typing import Dict, List, Optional, Any

def get_openai_api_key() -> str:
    """Get OpenAI API key from environment variable."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return api_key

def list_models() -> List[Dict[str, Any]]:
    """List all available models."""
    api_key = get_openai_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    response = requests.get("https://api.openai.com/v1/models", headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error listing models: {response.status_code} {response.text}")
    
    return response.json()["data"]

def retrieve_model(model_id: str) -> Dict[str, Any]:
    """Retrieve specific model information."""
    api_key = get_openai_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    response = requests.get(f"https://api.openai.com/v1/models/{model_id}", headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error retrieving model {model_id}: {response.status_code} {response.text}")
    
    return response.json()

def main():
    """Main function to list and display available models."""
    try:
        print("Fetching available models...")
        models = list_models()
        
        print(f"\nFound {len(models)} available models:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model['id']} (created: {model.get('created', 'N/A')}, owned by: {model.get('owned_by', 'N/A')})")
        
        # Save all models to file automatically
        with open("all_models.json", "w") as f:
            json.dump(models, f, indent=2)
        print(f"\nAll models saved to all_models.json")
        
        # Get details for some popular models
        popular_models = ['gpt-4o', 'gpt-4', 'gpt-3.5-turbo']
        for model_id in popular_models:
            try:
                model_exists = any(model['id'] == model_id for model in models)
                if model_exists:
                    print(f"\nFetching details for {model_id}...")
                    model_details = retrieve_model(model_id)
                    
                    # Save model details to file
                    filename = f"{model_id.replace(':', '_')}_details.json"
                    with open(filename, "w") as f:
                        json.dump(model_details, f, indent=2)
                    print(f"Model details saved to {filename}")
            except Exception as e:
                print(f"Error retrieving details for {model_id}: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()