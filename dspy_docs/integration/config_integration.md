# Integrating DSPy with AI-Scientist Configuration

This guide explains how to integrate DSPy with the AI-Scientist configuration system.

## Overview

AI-Scientist uses a configuration system based on the `configparser` module, loading from INI files.DSPy has its own configuration system through environment variables and the `dspy.settings` object.

## Integration Approach

### Adding DSPy Configuration to AI-Scientist

You can extend the existing `config.py` to include DSPy-specific settings:

```python
import configparser
import os
import dspy

def load_config():
    """Load configuration from files and set up DSPy"""
    config = configparser.ConfigParser()
    base_dir = os.path.dirname(os.path.dirname(__file__))
    config_rounds_path = os.path.join(base_dir, 'config_rounds.ini')
    config_gpu_path = os.path.join(base_dir, 'config_gpu.ini')
    config_dspy_path = os.path.join(base_dir, 'config_dspy.ini')
    
    # Load existing configs
    if os.path.exists(config_rounds_path):
        config.read(config_rounds_path)
    if os.path.exists(config_gpu_path):
        config.read(config_gpu_path)
        
    # Set up DSPy configuration
    if os.path.exists(config_dspy_path):
        config.read(config_dspy_path)
        
        # Configure DSPy caching
        if 'cache' in config['dspy']:
            cache_dir = config['dspy']['cache']
            os.environ['DSPY_CACHE_DIR'] = cache_dir
            
        # Configure DSPy LM
        if 'lm_provider' in config['dspy']:
            provider = config['dspy']['lm_provider']
            model = config['dspy']['lm_model']
            lm = dspy.LM(f'{provider}/{model}')
            dspy.configure(lm=lm)
    
    return config
```

### Example DSPy Configuration File (config_dspy.ini)

```ini
[dspy]
cache = ./dspy_cache
lm_provider = openai
lm_model = gpt-4o-mini
max_bootstrapped_demos = 3
max_labeled_demos = 5
```
