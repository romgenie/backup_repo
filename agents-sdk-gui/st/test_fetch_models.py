"""
Test script to demonstrate model fetching with mock data.
This version doesn't require an actual OpenAI API key.

Usage:
    python test_fetch_models.py
"""

import json
from datetime import datetime, timedelta

def get_mock_openai_models():
    """Generate mock data simulating OpenAI model responses."""
    # Current time for created timestamps
    now = datetime.now()
    
    # Mock model data
    mock_models = [
        # GPT-4 models
        {
            "id": "gpt-4o",
            "created": int((now - timedelta(days=30)).timestamp()),
            "owned_by": "openai",
            "object": "model"
        },
        {
            "id": "gpt-4o-mini",
            "created": int((now - timedelta(days=30)).timestamp()),
            "owned_by": "openai",
            "object": "model"
        },
        {
            "id": "gpt-4-turbo",
            "created": int((now - timedelta(days=120)).timestamp()),
            "owned_by": "openai",
            "object": "model"
        },
        {
            "id": "gpt-4",
            "created": int((now - timedelta(days=300)).timestamp()),
            "owned_by": "openai",
            "object": "model"
        },
        {
            "id": "gpt-4-vision-preview",
            "created": int((now - timedelta(days=200)).timestamp()),
            "owned_by": "openai",
            "object": "model"
        },
        {
            "id": "gpt-4-32k",
            "created": int((now - timedelta(days=350)).timestamp()),
            "owned_by": "openai",
            "object": "model"
        },
        
        # GPT-3.5 models
        {
            "id": "gpt-3.5-turbo",
            "created": int((now - timedelta(days=400)).timestamp()),
            "owned_by": "openai",
            "object": "model"
        },
        {
            "id": "gpt-3.5-turbo-16k",
            "created": int((now - timedelta(days=380)).timestamp()),
            "owned_by": "openai",
            "object": "model"
        },
        
        # Embedding models
        {
            "id": "text-embedding-3-small",
            "created": int((now - timedelta(days=90)).timestamp()),
            "owned_by": "openai",
            "object": "model"
        },
        {
            "id": "text-embedding-3-large",
            "created": int((now - timedelta(days=90)).timestamp()),
            "owned_by": "openai",
            "object": "model"
        },
        
        # Image models
        {
            "id": "dall-e-3",
            "created": int((now - timedelta(days=150)).timestamp()),
            "owned_by": "openai",
            "object": "model"
        },
        {
            "id": "dall-e-2",
            "created": int((now - timedelta(days=500)).timestamp()),
            "owned_by": "openai",
            "object": "model"
        },
        
        # Audio models
        {
            "id": "whisper-1",
            "created": int((now - timedelta(days=250)).timestamp()),
            "owned_by": "openai",
            "object": "model"
        },
    ]
    
    # Mock Models class with data property
    class MockModels:
        def __init__(self, data):
            self.data = data
            
    return MockModels(mock_models)

def fetch_models_demo():
    """Demonstrate model fetching with mock data."""
    try:
        print("Fetching models (using mock data)...")
        
        # Get mock models
        models = get_mock_openai_models()
        
        # Convert to dictionary for easier processing
        models_dict = {}
        for model in models.data:
            # Convert created timestamp to date
            created_date = datetime.fromtimestamp(model["created"]).strftime("%Y-%m-%d")
            
            models_dict[model["id"]] = {
                "id": model["id"],
                "created": created_date,
                "owned_by": model["owned_by"],
                "object": model["object"]
            }
        
        # Print number of models found
        print(f"\nFound {len(models_dict)} models")
        
        # Print formatted JSON for all models
        print("\nAll models:")
        print(json.dumps(models_dict, indent=2))
        
        # Filter for common models
        common_models = {k: v for k, v in models_dict.items() if any(
            name in k for name in [
                "gpt-4", "gpt-3.5", "dall-e", "whisper"
            ]
        )}
        
        # Print formatted JSON for common models
        print("\nCommon models:")
        print(json.dumps(common_models, indent=2))
        
        # Generate Python code for models_module.py
        print("\nPython code for models_module.py:")
        
        model_entries = []
        
        for model_id, model_info in common_models.items():
            # Skip embeddings models
            if "embedding" in model_id.lower():
                continue
                
            # Determine capabilities based on model name
            capabilities = ["text"]
            
            if "gpt-4" in model_id:
                capabilities.extend(["reasoning", "tools", "json_response"])
                if "vision" in model_id or "o" in model_id:
                    capabilities.extend(["images", "vision"])
            
            elif "gpt-3.5" in model_id:
                capabilities.extend(["tools", "json_response"])
            
            elif "dall-e" in model_id:
                capabilities = ["images", "image_generation"]
            
            elif "whisper" in model_id:
                capabilities = ["audio", "transcription"]
            
            # Determine max tokens based on model
            if "gpt-4" in model_id and "32k" in model_id:
                max_tokens = 32768
            elif "gpt-4" in model_id:
                max_tokens = 8192
            elif "gpt-3.5" in model_id and "16k" in model_id:
                max_tokens = 16384
            elif "gpt-3.5" in model_id:
                max_tokens = 4096
            else:
                max_tokens = 4096
                
            # Create description
            if "gpt-4" in model_id and "vision" in model_id:
                description = "GPT-4 with vision capabilities"
            elif "gpt-4o" in model_id:
                description = "GPT-4 Omni with multimodal capabilities"
            elif "gpt-4" in model_id and "turbo" in model_id:
                description = "More capable and up-to-date GPT-4 model"
            elif "gpt-4" in model_id:
                description = "Most capable GPT-4 model for complex tasks"
            elif "gpt-3.5" in model_id and "turbo" in model_id:
                description = "Fast and cost-effective model for most tasks"
            elif "dall-e" in model_id:
                description = "Image generation model"
            elif "whisper" in model_id:
                description = "Speech recognition model"
            else:
                description = f"OpenAI {model_id} model"
                
            # Format the model entry
            entry = f'    "{model_id}": {{\n'
            entry += f'        "provider": "OpenAI",\n'
            entry += f'        "capabilities": {capabilities},\n'
            entry += f'        "max_tokens": {max_tokens},\n'
            entry += f'        "created": "{model_info["created"]}",\n'
            entry += f'        "owned_by": "{model_info["owned_by"]}",\n'
            entry += f'        "description": "{description}"\n'
            entry += f'    }}'
            
            model_entries.append(entry)
        
        print("# Copy this section to your models_module.py initialization:")
        print("st.session_state.models = {")
        print(",\n".join(model_entries))
        print("}")
        
        # Also add anthropic models for completeness
        print("\n# Don't forget to also include Anthropic models:")
        
        anthropic_models = [
            {
                "id": "claude-3-opus-20240229",
                "created": "2024-02-29",
                "owned_by": "anthropic",
                "capabilities": ["text", "reasoning", "tools", "json_response", "images", "vision"],
                "max_tokens": 8192,
                "description": "Most capable Claude model for complex tasks"
            },
            {
                "id": "claude-3-sonnet-20240229",
                "created": "2024-02-29",
                "owned_by": "anthropic",
                "capabilities": ["text", "reasoning", "tools", "json_response", "images", "vision"],
                "max_tokens": 8192,
                "description": "Balanced Claude model for most use cases"
            },
            {
                "id": "claude-3-haiku-20240307",
                "created": "2024-03-07",
                "owned_by": "anthropic",
                "capabilities": ["text", "reasoning", "tools", "json_response", "images", "vision"],
                "max_tokens": 4096,
                "description": "Fast, efficient Claude model for simpler tasks"
            },
            {
                "id": "claude-3-5-sonnet-20240620",
                "created": "2024-06-20",
                "owned_by": "anthropic",
                "capabilities": ["text", "reasoning", "tools", "json_response", "images", "vision"],
                "max_tokens": 8192,
                "description": "Latest Claude model with strong reasoning capabilities"
            }
        ]
        
        anthropic_entries = []
        for model in anthropic_models:
            entry = f'    "{model["id"]}": {{\n'
            entry += f'        "provider": "Anthropic",\n'
            entry += f'        "capabilities": {model["capabilities"]},\n'
            entry += f'        "max_tokens": {model["max_tokens"]},\n'
            entry += f'        "created": "{model["created"]}",\n'
            entry += f'        "owned_by": "{model["owned_by"]}",\n'
            entry += f'        "description": "{model["description"]}"\n'
            entry += f'    }}'
            
            anthropic_entries.append(entry)
            
        for entry in anthropic_entries:
            print(entry + ",")
        
        return models_dict
    
    except Exception as e:
        print(f"Error in demo: {str(e)}")
        return None

if __name__ == "__main__":
    print("Running model fetching demonstration with mock data")
    print("(Note: This does not require an OpenAI API key)")
    print("-" * 70)
    
    fetch_models_demo()