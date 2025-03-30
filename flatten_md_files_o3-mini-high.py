#!/usr/bin/env python3
import os
import sys
import argparse
import re
import json
import curses

# Path for storing user preferences across sessions
PREFS_FILE = os.path.expanduser("~/.flatten_md_preferences.json")

def load_preferences():
    """Load saved directory selections from the preferences file."""
    if os.path.exists(PREFS_FILE):
        try:
            with open(PREFS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: could not load preferences: {e}")
            return {}
    else:
        return {}

def save_preferences(prefs):
    """Save directory selections to the preferences file."""
    try:
        with open(PREFS_FILE, 'w') as f:
            json.dump(prefs, f, indent=2)
    except Exception as e:
        print(f"Warning: could not save preferences: {e}")

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

def flatten_md_files(directory_path, output_file=None):
    """
    Recursively reads all .md files in a directory and combines them into a single file.
    
    Args:
        directory_path: Path to the directory containing markdown files
        output_file: Path to the output file (default: derived from directory structure)
    """
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory")
        return
    
    if output_file is None:
        output_file = get_default_output_filename(directory_path)
    
    all_content = []
    
    # Walk through the directory
    for root, dirs, files in os.walk(directory_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
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
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    # Write combined content to output file
    with open(output_file, 'w', encoding='utf-8') as output:
        output.write(f"# Flattened Markdown Documents from {directory_path}\n\n")
        output.write("".join(all_content))
    
    print(f"Created {output_file} with content from {len(all_content)} markdown files")

def interactive_menu(starting_dir):
    """
    Launch an interactive terminal menu (using curses) that lets the user navigate
    through directories starting from `starting_dir` and select/deselect them.
    
    The user's selections are saved in a preferences file so that they persist across sessions.
    
    Returns:
        list: Absolute paths of selected directories.
    """
    # Load saved preferences
    prefs = load_preferences()
    company, repo = extract_company_repo(starting_dir)
    if company and repo:
        key = f"{company}/{repo}"
    else:
        key = starting_dir
    # global_selection holds relative paths (relative to starting_dir) that are marked
    global_selection = set(prefs.get(key, []))
    
    def menu_loop(stdscr):
        curses.curs_set(0)  # Hide the cursor
        current_rel_path = ""  # Relative path from starting_dir
        current_idx = 0
        
        while True:
            abs_current_dir = os.path.join(starting_dir, current_rel_path)
            try:
                # List only directories (skip hidden)
                entries = sorted(
                    [d for d in os.listdir(abs_current_dir)
                     if os.path.isdir(os.path.join(abs_current_dir, d)) and not d.startswith('.')]
                )
            except Exception as e:
                entries = []
            
            stdscr.clear()
            # Display current location (breadcrumb)
            display_path = os.path.join(starting_dir, current_rel_path) if current_rel_path else starting_dir
            stdscr.addstr(0, 0, f"Current Directory: {display_path}")
            stdscr.addstr(1, 0, "↑/↓: Navigate | Enter: Open | Space: Toggle select | Backspace: Up | q: Finish")
            
            # Show directory entries with selection markers
            for i, entry in enumerate(entries):
                # Compute the relative path of the entry (from starting_dir)
                rel_entry = os.path.join(current_rel_path, entry) if current_rel_path else entry
                marker = "[x]" if rel_entry in global_selection else "[ ]"
                line = f"  {marker} {entry}"
                if i == current_idx:
                    stdscr.addstr(i + 3, 0, "> " + line, curses.A_REVERSE)
                else:
                    stdscr.addstr(i + 3, 0, "  " + line)
            
            stdscr.refresh()
            key = stdscr.getch()
            
            if key == curses.KEY_UP:
                if entries:
                    current_idx = (current_idx - 1) % len(entries)
            elif key == curses.KEY_DOWN:
                if entries:
                    current_idx = (current_idx + 1) % len(entries)
            elif key in [curses.KEY_ENTER, 10, 13]:
                # Enter: navigate into the selected directory (if available)
                if entries:
                    selected_entry = entries[current_idx]
                    new_rel = os.path.join(current_rel_path, selected_entry) if current_rel_path else selected_entry
                    # Navigate into the directory and reset selection index
                    current_rel_path = new_rel
                    current_idx = 0
            elif key in [curses.KEY_BACKSPACE, 127, 8]:
                # Backspace: go up one directory if possible
                if current_rel_path:
                    current_rel_path = os.path.dirname(current_rel_path)
                    current_idx = 0
            elif key == ord(' '):
                # Space: toggle selection for the currently highlighted directory
                if entries:
                    selected_entry = entries[current_idx]
                    rel_entry = os.path.join(current_rel_path, selected_entry) if current_rel_path else selected_entry
                    if rel_entry in global_selection:
                        global_selection.remove(rel_entry)
                    else:
                        global_selection.add(rel_entry)
            elif key == ord('q'):
                # Quit the menu and finish selection
                break
        
        return global_selection

    # Run the curses menu loop
    selected = curses.wrapper(menu_loop)
    # Save the user's selections in the preferences file (using the company/repo or base directory as key)
    prefs[key] = list(selected)
    save_preferences(prefs)
    # Convert relative paths to absolute paths before returning
    selected_abs = [os.path.join(starting_dir, rel) for rel in selected]
    return selected_abs

def main():
    parser = argparse.ArgumentParser(description='Flatten markdown files into a single document')
    # In --menu mode the directory argument is optional; otherwise it is required.
    parser.add_argument('directory', nargs='?', help='Directory containing markdown files or starting directory for menu')
    parser.add_argument('-o', '--output', 
                        help='Output filename (default is derived from directory path)')
    parser.add_argument('--menu', action='store_true', help='Launch interactive directory selection menu')
    
    args = parser.parse_args()
    
    if args.menu:
        # Use provided directory as the starting point, or default to current directory.
        starting_dir = args.directory if args.directory else os.getcwd()
        selected_dirs = interactive_menu(starting_dir)
        if not selected_dirs:
            print("No directories selected. Exiting.")
            sys.exit(0)
        for directory in selected_dirs:
            print(f"\nProcessing directory: {directory}")
            # For each selected directory, determine the output filename.
            output_file = args.output if args.output else get_default_output_filename(directory)
            flatten_md_files(directory, output_file)
    else:
        # If not in menu mode, require a directory argument.
        if not args.directory:
            parser.error("the following argument is required: directory")
        flatten_md_files(args.directory, args.output)

if __name__ == "__main__":
    main()
