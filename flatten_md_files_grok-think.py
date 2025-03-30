#!/usr/bin/env python3
import os
import sys
import argparse
import re
import json
import inquirer

def extract_company_repo_with_root(directory_path):
    """
    Extract company, repository name, and repo root from path.
    
    Example: /Users/timgregg/mcp/Github/stanfordnlp/dspy/docs/docs
    Returns: ('stanfordnlp', 'dspy', '/Users/timgregg/mcp/Github/stanfordnlp/dspy')
    """
    pattern = r'(?:\/Github\/)([^\/]+)\/([^\/]+)'
    match = re.search(pattern, directory_path)
    if match:
        company = match.group(1)
        repo = match.group(2)
        repo_root = directory_path[:match.end(0)]
        return company, repo, repo_root
    # Fallback: Use last two directories and directory_path as root
    parts = [p for p in directory_path.split(os.sep) if p and not p.startswith('.')]
    if len(parts) >= 2:
        company = parts[-2]
        repo = parts[-1]
        repo_root = os.path.join('/', *parts[:-1], repo)
        return company, repo, repo_root
    return None, None, directory_path

def get_default_output_filename(directory_path):
    """Generate default output filename based on company and repository names."""
    company, repo, _ = extract_company_repo_with_root(directory_path)
    if company and repo:
        return f"{company}_{repo}.md"
    return "flattened_docs.md"

def get_all_subdirs(directory_path):
    """
    Get all subdirectories under directory_path, relative to directory_path.
    Excludes hidden directories.
    """
    subdirs = []
    for root, dirs, _ in os.walk(directory_path):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for d in dirs:
            full_path = os.path.join(root, d)
            rel_path = os.path.relpath(full_path, directory_path)
            subdirs.append(rel_path)
    return sorted(subdirs)

CONFIG_FILE = os.path.expanduser('~/.flatten_md_config.json')

def load_config():
    """Load configuration from JSON file."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_config(config):
    """Save configuration to JSON file."""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

def flatten_md_files(directory_path, output_file=None, selected_dirs=None, repo_root=None):
    """
    Recursively reads selected .md files and combines them into a single file.
    
    Args:
        directory_path: Path to the directory containing markdown files
        output_file: Path to the output file (default: derived from directory structure)
        selected_dirs: List of directories relative to repo_root to include
        repo_root: Root of the repository for relative path calculations
    """
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory")
        return
    
    if output_file is None:
        output_file = get_default_output_filename(directory_path)
    
    if selected_dirs is None:
        selected_dirs = []
    if repo_root is None:
        repo_root = directory_path

    all_content = []
    file_count = 0

    for root, dirs, files in os.walk(directory_path):
        if selected_dirs:
            rel_root = os.path.relpath(root, repo_root)
            is_included = any(rel_root == selected or rel_root.startswith(selected + os.sep) 
                            for selected in selected_dirs)
            if is_included:
                for filename in files:
                    if filename.endswith('.md') and not filename.startswith('.'):
                        file_path = os.path.join(root, filename)
                        relative_path = os.path.relpath(file_path, directory_path)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as file:
                                content = file.read()
                            file_content = f"\n\n## File: {relative_path}\n\n{content}\n\n## End of {relative_path}\n\n---\n"
                            all_content.append(file_content)
                            file_count += 1
                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")
            # Filter subdirectories
            dirs[:] = [d for d in dirs if any(os.path.join(rel_root, d) == selected or 
                                            os.path.join(rel_root, d).startswith(selected + os.sep) 
                                            for selected in selected_dirs)]
        else:
            # Include all .md files
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for filename in files:
                if filename.endswith('.md') and not filename.startswith('.'):
                    file_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(file_path, directory_path)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            content = file.read()
                        file_content = f"\n\n## File: {relative_path}\n\n{content}\n\n## End of {relative_path}\n\n---\n"
                        all_content.append(file_content)
                        file_count += 1
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

    if all_content:
        with open(output_file, 'w', encoding='utf-8') as output:
            output.write(f"# Flattened Markdown Documents from {directory_path}\n\n")
            output.write("".join(all_content))
        print(f"Created {output_file} with content from {file_count} markdown files")
    else:
        print(f"No markdown files processed for {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Flatten markdown files into a single document')
    parser.add_argument('directory', help='Directory containing markdown files to flatten')
    parser.add_argument('-o', '--output', help='Output filename (default: company_repo.md)')
    parser.add_argument('--menu', action='store_true', 
                       help='Show interactive menu to select directories')
    
    args = parser.parse_args()

    # Extract company, repo, and repo root
    company, repo, repo_root = extract_company_repo_with_root(args.directory)
    if not repo_root:
        print("Could not determine repository root")
        sys.exit(1)

    # Load existing selections
    config = load_config()
    key = f"{company}/{repo}"
    selections = config.get(key, [])

    if args.menu:
        # Get all subdirectories under the specified directory
        all_subdirs = get_all_subdirs(args.directory)
        if not all_subdirs:
            print(f"No subdirectories found under {args.directory}")
            all_subdirs = ['.']
        
        rel_dir = os.path.relpath(args.directory, repo_root)
        pre_selected = [d for d in all_subdirs if os.path.join(rel_dir, d) in selections]
        
        questions = [
            inquirer.Checkbox('selected',
                            message=f"Select directories to include under {args.directory} (selecting a directory includes all its subdirectories)",
                            choices=all_subdirs,
                            default=pre_selected),
        ]
        
        answers = inquirer.prompt(questions)
        if answers is None:
            print("Menu cancelled")
            sys.exit(0)
        
        selected_subdirs = answers['selected']
        selected_dirs = [os.path.join(rel_dir, d) if d != '.' else rel_dir 
                        for d in selected_subdirs]
        
        # Save selections
        config[key] = selected_dirs
        save_config(config)
    else:
        selected_dirs = selections

    # Flatten files with selections
    flatten_md_files(args.directory, args.output, selected_dirs, repo_root)

if __name__ == "__main__":
    main()