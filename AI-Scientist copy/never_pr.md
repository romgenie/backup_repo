# Local Development Changes - DO NOT PR

This file documents local changes that should not be included in future pull requests.

## Abstract Length Increase

The default maximum abstract length in `generate_ideas.py` has been increased from 1000 to 3000 characters:

```python
# Changed in extract_info_from_work function
def extract_info_from_work(work: Work, max_abstract_length: int = 3000) -> dict[str, str]:
```

## Config Files for Settings

Added a configuration system to control various settings from configuration files.

Files created/modified:
- `/config_rounds.ini` - Configuration for iteration settings
- `/config_gpu.ini` - Configuration for GPU/device settings
- `/ai_scientist/config.py` - Utility module to load configuration

The config_rounds.ini includes settings for:
```ini
[idea_generation]
num_reflections = 3
max_num_generations = 20
max_attempts = 10
max_num_iterations = 10

[experiments]
max_iters = 4
max_runs = 5

[review]
num_reflections = 5
num_reviews_ensemble = 5

[writeup]
num_cite_rounds = 20
num_error_corrections = 5
```

The config_gpu.ini includes settings for:
```ini
[gpu]
device = "mps"
#device = "cuda"
```

## How to maintain these changes

When pulling changes from remote, you may need to:

1. Check if your local changes to `max_abstract_length` and config system are overwritten
2. Re-apply the changes if necessary
3. Consider using `git stash` before pulling and `git stash apply` after pulling to preserve these changes

## MPS Support for Apple Silicon

Several modifications were made to better support MPS (Metal Performance Shaders) on Apple Silicon:

1. Added `config_gpu.ini` for centralized device configuration:
   - Created a configuration file to control which device ("mps" or "cuda") is used
   - Updated config.py to load this configuration file

2. Updated `templates/nanoGPT/experiment.py` and `templates/nanoGPT_lite/experiment.py`:
   - Modified device selection to read from config_gpu.ini
   - Modified GradScaler initialization to only be enabled on CUDA with float16, not on MPS
   - Added condition: `scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16" and device == "cuda"))`

3. Updated `launch_scientist.py`:
   - Modified `get_available_gpus()` to consider the device preference from config
   - Updated main function to load the configuration and use the selected device
   - Updated `worker()` function to handle MPS devices, which don't use CUDA_VISIBLE_DEVICES

These changes allow the code to run properly on both NVIDIA GPUs and Apple Silicon machines by simply changing the configuration file without needing to manually adjust settings in code files.

## How to avoid committing these changes

When committing, be careful not to include:
- config_rounds.ini
- config_gpu.ini
- ai_scientist/config.py 
- Changes to extract_info_from_work function in generate_ideas.py
- MPS-related changes in templates/nanoGPT/experiment.py, templates/nanoGPT_lite/experiment.py and launch_scientist.py

You can use:
```bash
git reset -- config_rounds.ini config_gpu.ini ai_scientist/config.py templates/nanoGPT/experiment.py templates/nanoGPT_lite/experiment.py launch_scientist.py
```

Or add these files to your personal .git/info/exclude file:
```
config_rounds.ini
config_gpu.ini
ai_scientist/config.py
templates/nanoGPT/experiment.py
templates/nanoGPT_lite/experiment.py
launch_scientist.py
```