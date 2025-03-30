#!/usr/bin/env python
"""
Agent Runner Launcher

This script serves as an entry point for launching different components of
the Agent Runner application. It provides a simple CLI interface to run
the demo, dashboard, or model management UI.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

def get_script_paths():
    """Get paths to available scripts."""
    current_dir = Path(__file__).parent.absolute()
    
    scripts = {
        "demo": current_dir / "demo.py",
        "dashboard": current_dir / "dashboard.py",
        "model": current_dir / "model_management.py",
        "selector": current_dir / "model_selector.py",
    }
    
    # Verify scripts exist
    for name, path in scripts.items():
        if not path.exists():
            print(f"Warning: {name} script not found at {path}")
    
    return scripts

def launch_streamlit(script_path, port=None):
    """Launch a Streamlit script with specified port."""
    cmd = ["streamlit", "run", str(script_path)]
    
    if port:
        cmd.extend(["--server.port", str(port)])
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStreamlit process terminated.")

def main():
    """Main entry point for the launcher."""
    parser = argparse.ArgumentParser(description="Agent Runner Launcher")
    
    # Command argument
    parser.add_argument(
        "command",
        choices=["demo", "dashboard", "model", "all"],
        help="Component to launch: demo, dashboard, model, or all"
    )
    
    # Optional port argument
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8501,
        help="Port to run Streamlit on (default: 8501)"
    )
    
    args = parser.parse_args()
    
    # Get script paths
    scripts = get_script_paths()
    
    # Launch requested component
    if args.command == "demo":
        print("Launching Agent Runner Demo...")
        launch_streamlit(scripts["demo"], args.port)
    
    elif args.command == "dashboard":
        print("Launching Agent Runner Dashboard...")
        launch_streamlit(scripts["dashboard"], args.port)
    
    elif args.command == "model":
        print("Launching Model Management UI...")
        launch_streamlit(scripts["model"], args.port)
    
    elif args.command == "all":
        print("Launch option 'all' is not yet implemented.")
        print("Please launch individual components.")
        print("Available commands:")
        print("  python run.py demo")
        print("  python run.py dashboard")
        print("  python run.py model")
        sys.exit(1)

if __name__ == "__main__":
    main()