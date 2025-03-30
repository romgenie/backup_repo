import configparser
import os

def load_config():
    """
    Load the configuration from config_rounds.ini and config_gpu.ini files
    """
    config = configparser.ConfigParser()
    base_dir = os.path.dirname(os.path.dirname(__file__))
    config_rounds_path = os.path.join(base_dir, 'config_rounds.ini')
    config_gpu_path = os.path.join(base_dir, 'config_gpu.ini')
    
    # Set default values if config files don't exist
    if not os.path.exists(config_rounds_path):
        config['idea_generation'] = {
            'num_reflections': '3',
            'max_num_generations': '20',
            'max_attempts': '10',
            'max_num_iterations': '10'
        }
        config['experiments'] = {
            'max_iters': '4',
            'max_runs': '5'
        }
        config['review'] = {
            'num_reflections': '5',
            'num_reviews_ensemble': '5'
        }
        config['writeup'] = {
            'num_cite_rounds': '20',
            'num_error_corrections': '5'
        }
    else:
        config.read(config_rounds_path)
    
    # Load GPU config if it exists
    if os.path.exists(config_gpu_path):
        config.read(config_gpu_path)  # This will override any duplicate sections/keys
    
    return config