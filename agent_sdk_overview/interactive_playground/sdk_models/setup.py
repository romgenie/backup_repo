"""
Setup script for installing required dependencies for the enhanced SDK features.
Run this script to ensure all dependencies are installed.
"""
import subprocess
import sys
import os

# Define dependencies
DEPENDENCIES = [
    "jsonschema",
    "streamlit-mermaid",
    "streamlit>=1.20.0"
]

def install_dependencies():
    """Install required dependencies for the enhanced SDK features."""
    print("Installing dependencies for the enhanced SDK features...")
    
    for package in DEPENDENCIES:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")
            return False
    
    print("All dependencies installed successfully!")
    return True

if __name__ == "__main__":
    if install_dependencies():
        print("\nSetup complete. You can now run the application with:")
        print("  streamlit run main.py")
    else:
        print("\nSetup failed. Please check the error messages above.")