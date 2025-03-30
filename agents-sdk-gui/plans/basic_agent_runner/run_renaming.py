#!/usr/bin/env python3
"""
Master script to run all renaming scripts in sequence.

This script:
1. Runs each renaming script in the proper order
2. Collects and combines their reports
3. Creates a summary of all changes
"""

import os
import sys
import subprocess
from datetime import datetime

# The order matters - we want to rename classes first, then functions, then variables, then UI
RENAMING_SCRIPTS = [
    "rename_tool_classes.py",
    "rename_tool_functions.py",
    "rename_tool_variables.py", 
    "rename_ui_components.py"
]

def run_script(script_path, root_dir, auto_confirm=False):
    """Run a renaming script with the given root directory."""
    cmd = [sys.executable, script_path, root_dir]
    
    # For automated runs, we need to handle the confirmation prompt
    if auto_confirm:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        stdout, stderr = process.communicate(input="y\n")
        return stdout, stderr, process.returncode
    else:
        # For interactive runs, just let the script handle its own I/O
        result = subprocess.run(cmd)
        return None, None, result.returncode

def combine_reports(report_files, output_file):
    """Combine multiple Markdown reports into a single report."""
    combined = "# Combined Renaming Report\n\n"
    combined += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    total_files_modified = set()
    total_replacements = 0
    
    for report_file in report_files:
        if not os.path.exists(report_file):
            continue
            
        with open(report_file, 'r') as f:
            content = f.read()
        
        # Extract the report name from filename
        report_name = os.path.basename(report_file).replace("_report.md", "").replace("_", " ").title()
        combined += f"## {report_name}\n\n"
        
        # Find the files section
        files_section = content.split("## Modified")[1].split("\n\n")[1] if "## Modified" in content else ""
        files = [line.strip("- \n") for line in files_section.split("\n") if line.startswith("-")]
        total_files_modified.update(files)
        
        # Find the replacements section
        replacements_section = content.split("## Replacements")[1] if "## Replacements" in content else ""
        replacements = [line.strip("- \n") for line in replacements_section.split("\n") if line.startswith("-")]
        
        # Count replacements
        replacement_count = sum(int(r.split(": ")[1].split(" ")[0]) for r in replacements if ": " in r)
        total_replacements += replacement_count
        
        # Add content to combined report, excluding the original headers
        sections = content.split("\n\n")[2:]  # Skip the title and first blank line
        combined += "\n\n".join(sections) + "\n\n"
    
    # Add summary section
    combined += "# Overall Summary\n\n"
    combined += f"- Total files modified: {len(total_files_modified)}\n"
    combined += f"- Total replacements made: {total_replacements}\n\n"
    combined += "## All Modified Files\n\n"
    for file in sorted(total_files_modified):
        combined += f"- {file}\n"
    
    with open(output_file, 'w') as f:
        f.write(combined)
    
    return len(total_files_modified), total_replacements

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_renaming.py <root_directory> [--auto]")
        sys.exit(1)
    
    root_dir = sys.argv[1]
    auto_confirm = "--auto" in sys.argv
    
    print(f"{'Automatic' if auto_confirm else 'Interactive'} renaming process starting...")
    print(f"Target directory: {root_dir}")
    
    # Make sure the scripts exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for script in RENAMING_SCRIPTS:
        script_path = os.path.join(current_dir, script)
        if not os.path.exists(script_path):
            print(f"Error: Script {script} not found at {script_path}")
            sys.exit(1)
    
    if not auto_confirm:
        print("\nThis will run the following renaming scripts in sequence:")
        for i, script in enumerate(RENAMING_SCRIPTS, 1):
            print(f"{i}. {script}")
        
        confirm = input("\nProceed with the renaming process? (y/n): ").lower()
        if confirm != 'y':
            print("Renaming process cancelled.")
            sys.exit(0)
    
    # Run each script
    for i, script in enumerate(RENAMING_SCRIPTS, 1):
        script_path = os.path.join(current_dir, script)
        print(f"\n{'='*50}")
        print(f"Running script {i}/{len(RENAMING_SCRIPTS)}: {script}")
        print(f"{'='*50}\n")
        
        stdout, stderr, return_code = run_script(script_path, root_dir, auto_confirm)
        
        if return_code != 0:
            print(f"Error running {script}. Return code: {return_code}")
            if stderr:
                print(f"Error output:\n{stderr}")
            
            if not auto_confirm:
                choice = input("\nContinue with the next script? (y/n): ").lower()
                if choice != 'y':
                    print("Renaming process halted.")
                    sys.exit(1)
    
    # Combine reports
    report_files = [
        os.path.join(current_dir, "class_renaming_report.md"),
        os.path.join(current_dir, "function_renaming_report.md"),
        os.path.join(current_dir, "variable_renaming_report.md"),
        os.path.join(current_dir, "ui_component_renaming_report.md")
    ]
    
    output_file = os.path.join(current_dir, "combined_renaming_report.md")
    total_files, total_replacements = combine_reports(report_files, output_file)
    
    print(f"\n{'='*50}")
    print("Renaming process completed!")
    print(f"{'='*50}")
    print(f"\nTotal files modified: {total_files}")
    print(f"Total replacements made: {total_replacements}")
    print(f"\nCombined report saved to: {output_file}")
    
    print("\nNext steps:")
    print("1. Review the combined report to verify all changes")
    print("2. Run tests to ensure code functionality is maintained")
    print("3. Update documentation to reflect the new naming conventions")

if __name__ == "__main__":
    main()