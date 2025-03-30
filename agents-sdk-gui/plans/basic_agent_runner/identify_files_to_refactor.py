#!/usr/bin/env python3

import os
import sys

def count_lines(file_path):
    """Count the number of lines in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return 0

def find_large_py_files(directory, min_lines):
    """Find Python files with more than min_lines lines."""
    large_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                line_count = count_lines(file_path)
                
                if line_count > min_lines:
                    large_files.append((file_path, line_count))
    
    return large_files

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <directory> <min_lines>", file=sys.stderr)
        sys.exit(1)
    
    directory = sys.argv[1]
    try:
        min_lines = int(sys.argv[2])
    except ValueError:
        print("Error: min_lines must be an integer", file=sys.stderr)
        sys.exit(1)
    
    large_files = find_large_py_files(directory, min_lines)
    
    # Sort by line count in descending order
    large_files.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Python files with more than {min_lines} lines in {directory}:")
    for file_path, line_count in large_files:
        print(f"{line_count} lines: {file_path}")

if __name__ == "__main__":
    main()