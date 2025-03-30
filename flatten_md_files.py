#!/usr/bin/env python3
import os
import sys
import argparse
import re
import json
import shutil
from pathlib import Path

try:
    from simple_term_menu import TerminalMenu
    from colorama import Fore, Style, init
    init()  # Initialize colorama
except ImportError:
    print("Required packages not found. Installing simple-term-menu and colorama...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "simple-term-menu", "colorama"])
    from simple_term_menu import TerminalMenu
    from colorama import Fore, Style, init
    init()  # Initialize colorama

def extract_company_repo(directory_path):
    """
    Extract company and repository name from path.
    
    Example: /Users/timgregg/mcp/Github/stanfordnlp/dspy/docs/docs
    Company: stanfordnlp
    Repository: dspy
    
    Returns:
        tuple: (company, repo) or (None, None) if pattern not found
    """
    # Try to extract a path pattern like /Github/company/repo/
    pattern = r'(?:\/Github\/)([^\/]+)\/([^\/]+)'
    match = re.search(pattern, directory_path)
    
    if match:
        company = match.group(1)
        repo = match.group(2)
        return company, repo
    
    # If specific pattern not found, try to extract the last two meaningful directories
    parts = [p for p in directory_path.split(os.sep) if p and not p.startswith('.')]
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    
    return None, None

def get_default_output_filename(directory_path):
    """Generate default output filename based on company and repository names."""
    company, repo = extract_company_repo(directory_path)
    
    if company and repo:
        return f"{company}_{repo}.md"
    else:
        return "flattened_docs.md"

def get_preferences_path():
    """Get the path to the preferences file."""
    # Use a local config directory in the script's location
    script_dir = Path(__file__).resolve().parent
    config_dir = script_dir / ".md_flattener"
    
    try:
        config_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # Fallback to a temp directory if home directory isn't accessible
        import tempfile
        config_dir = Path(tempfile.gettempdir()) / "md_flattener"
        config_dir.mkdir(parents=True, exist_ok=True)
        
    return config_dir / "preferences.json"

def load_preferences():
    """Load saved directory preferences."""
    prefs_path = get_preferences_path()
    if prefs_path.exists():
        try:
            with open(prefs_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_preferences(preferences):
    """Save directory preferences."""
    prefs_path = get_preferences_path()
    with open(prefs_path, 'w') as f:
        json.dump(preferences, f, indent=2)

def directory_menu(base_path):
    """
    Interactive directory selection menu.
    
    Args:
        base_path: Base directory to start from
        
    Returns:
        List of selected directories
    """
    # Load previous preferences
    preferences = load_preferences()
    company, repo = extract_company_repo(base_path)
    pref_key = f"{company}_{repo}" if company and repo else str(base_path)
    
    # Get previously selected dirs, if any
    selected_dirs = set(preferences.get(pref_key, []))
    
    # Get all valid subdirectories recursively
    all_dirs = []
    for root, dirs, _ in os.walk(base_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            all_dirs.append(dir_path)
    
    # Sort directories by path
    all_dirs.sort()
    
    def display_menu():
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Display title
        terminal_width = shutil.get_terminal_size().columns
        title = f" Directory Selection for {base_path} "
        padding = "=" * ((terminal_width - len(title)) // 2)
        print(f"{Fore.CYAN}{padding}{title}{padding}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Navigate and select directories to include in the flattened document.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Space to select/deselect, Enter to confirm.{Style.RESET_ALL}\n")
        
        # Create menu items with visual indicators
        menu_items = []
        for dir_path in all_dirs:
            rel_path = os.path.relpath(dir_path, base_path)
            depth = len(Path(rel_path).parts) - 1
            indent = "  " * depth
            prefix = f"{Fore.GREEN}[✓]{Style.RESET_ALL} " if dir_path in selected_dirs else f"{Fore.RED}[ ]{Style.RESET_ALL} "
            menu_items.append(f"{prefix}{indent}{os.path.basename(dir_path)}")
        
        menu_items.append(f"\n{Fore.BLUE}✓ Select All{Style.RESET_ALL}")
        menu_items.append(f"{Fore.BLUE}✗ Deselect All{Style.RESET_ALL}")
        menu_items.append(f"{Fore.BLUE}✓ Done{Style.RESET_ALL}")
        
        return menu_items, all_dirs
    
    menu_items, all_dirs = display_menu()
    terminal_menu = TerminalMenu(
        menu_items,
        title="",
        cycle_cursor=True,
        clear_screen=False,
        skip_empty_entries=True
    )
    
    while True:
        selection = terminal_menu.show()
        
        if selection is None:
            # User pressed Esc or Ctrl+C
            return list(selected_dirs)
        
        # Check if we're at one of the special options at the bottom
        select_all_idx = len(all_dirs)
        deselect_all_idx = len(all_dirs) + 1
        done_idx = len(all_dirs) + 2
        
        if selection == select_all_idx:
            # Select All
            selected_dirs = set(all_dirs)
        elif selection == deselect_all_idx:
            # Deselect All
            selected_dirs = set()
        elif selection == done_idx:
            # Done
            break
        else:
            # Toggle selection for directory
            dir_path = all_dirs[selection]
            if dir_path in selected_dirs:
                selected_dirs.remove(dir_path)
            else:
                selected_dirs.add(dir_path)
        
        # Update menu
        menu_items, all_dirs = display_menu()
        terminal_menu = TerminalMenu(
            menu_items,
            title="",
            cycle_cursor=True,
            clear_screen=False,
            skip_empty_entries=True
        )
    
    # Save preferences
    preferences[pref_key] = list(selected_dirs)
    save_preferences(preferences)
    
    return list(selected_dirs)

def flatten_md_files(directory_path, output_file=None, selected_dirs=None):
    """
    Recursively reads all .md files in a directory and combines them into a single file.
    
    Args:
        directory_path: Path to the directory containing markdown files
        output_file: Path to the output file (default: derived from directory structure)
        selected_dirs: List of specific directories to include (if None, include all)
    """
    if not os.path.isdir(directory_path):
        print(f"{Fore.RED}Error: {directory_path} is not a valid directory{Style.RESET_ALL}")
        return
    
    if output_file is None:
        output_file = get_default_output_filename(directory_path)
    
    all_content = []
    file_count = 0
    
    # Walk through the directory
    for root, dirs, files in os.walk(directory_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        # If selected_dirs is provided, check if this directory should be included
        if selected_dirs is not None:
            if root != directory_path and root not in selected_dirs:
                # Skip directories that aren't selected
                dirs[:] = []
                continue
        
        # Process only .md files, skip hidden files
        md_files = [f for f in files if f.endswith('.md') and not f.startswith('.')]
        
        for filename in md_files:
            file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(file_path, directory_path)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # Add header and footer for each file
                file_content = f"\n\n## File: {relative_path}\n\n"
                file_content += content
                file_content += f"\n\n## End of {relative_path}\n\n"
                file_content += "---\n"
                
                all_content.append(file_content)
                file_count += 1
            except Exception as e:
                print(f"{Fore.RED}Error reading {file_path}: {e}{Style.RESET_ALL}")
    
    if file_count == 0:
        print(f"{Fore.YELLOW}Warning: No markdown files found in the selected directories.{Style.RESET_ALL}")
        return
    
    # Write combined content to output file
    with open(output_file, 'w', encoding='utf-8') as output:
        output.write(f"# Flattened Markdown Documents from {directory_path}\n\n")
        output.write("".join(all_content))
    
    print(f"{Fore.GREEN}Created {output_file} with content from {file_count} markdown files{Style.RESET_ALL}")

def main():
    parser = argparse.ArgumentParser(description='Flatten markdown files into a single document')
    parser.add_argument('directory', help='Directory containing markdown files to flatten')
    
    # Get default output filename for help message
    default_filename = "company_repo.md (derived from directory path)"
    
    parser.add_argument('-o', '--output', 
                        help=f'Output filename (default: {default_filename})')
    parser.add_argument('--menu', action='store_true',
                        help='Show an interactive directory selection menu')
    
    args = parser.parse_args()
    
    if args.menu:
        selected_dirs = directory_menu(args.directory)
        flatten_md_files(args.directory, args.output, selected_dirs)
    else:
        flatten_md_files(args.directory, args.output)

if __name__ == "__main__":
    main()