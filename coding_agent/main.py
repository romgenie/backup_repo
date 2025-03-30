"""
coding_agent.py

A single-file agent called "coding_agent" that performs various coding tasks,
including file and directory operations, code execution, and multiple modular workflows.
It enforces guardrails to prevent unsafe file writes and validates outputs
to ensure no incomplete code is produced.
All file write operations and code executions are traced and logged for auditing.
"""

import os
import datetime
import logging
import json
import argparse
import io
import contextlib
from agents import Agent, Runner, function_tool

# Configure logging for tracing file write and code execution operations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# The allowed base directory is the directory from which the agent is launched.
BASE_DIR = os.getcwd()

def is_within_base_directory(path: str) -> bool:
    """Check if the given path is within the BASE_DIR."""
    abs_path = os.path.abspath(path)
    return abs_path.startswith(BASE_DIR)

# ---------------------------------------------------------------------
# Tool Definitions: File and Directory Operations
# ---------------------------------------------------------------------

@function_tool
def read_file(path: str) -> str:
    """
    Reads the contents of a file given its path.

    Args:
        path: The path to the file.
    Returns:
        The file content, or an error message if the file doesn't exist.
    """
    if not os.path.exists(path):
        return f"Error: File '{path}' not found."
    try:
        with open(path, 'r') as f:
            contents = f.read()
        return contents
    except Exception as e:
        return f"Error reading file '{path}': {e}"

@function_tool
def write_file(path: str, content: str) -> str:
    """
    Writes content to a file given its path.
    Guardrail: Only allows writing to files within the BASE_DIR.
    Tracing: Logs every file write operation with details.
    Enhancement: Automatically creates the target directory if it does not exist.

    Args:
        path: The target file path.
        content: The content to write.
    Returns:
        A success message or a guardrail rejection message.
    """
    if not is_within_base_directory(path):
        return f"Guardrail Triggered: Write operation to '{path}' is not allowed (outside {BASE_DIR})."
    
    try:
        # Ensure the directory exists; if not, create it.
        dir_path = os.path.dirname(os.path.abspath(path))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Directory created: '{dir_path}'.")
        
        with open(path, 'w') as f:
            f.write(content)
        timestamp = datetime.datetime.now().isoformat()
        logger.info(f"File written: '{path}' at {timestamp} (wrote {len(content)} characters).")
        return f"File '{os.path.basename(path)}' has been written successfully with the provided content."
    except Exception as e:
        return f"Error writing file '{path}': {e}"

@function_tool
def list_directory(path: str) -> list[str]:
    """
    Lists the contents of a directory.
    If an empty string is provided, the BASE_DIR is used.

    Args:
        path: The directory path. Provide an empty string to default to BASE_DIR.
    Returns:
        A list of directory contents or an error message if the path is invalid.
    """
    if not path:
        path = BASE_DIR
    if not os.path.isdir(path):
        return [f"Error: '{path}' is not a valid directory."]
    try:
        return os.listdir(path)
    except Exception as e:
        return [f"Error listing directory '{path}': {e}"]

# ---------------------------------------------------------------------
# Tool for Code Execution
# ---------------------------------------------------------------------

@function_tool
def execute_code(code: str) -> str:
    """
    Executes the given Python code and returns a structured JSON containing details of the execution.
    The returned JSON includes:
      - execution_name: A name for this code execution (set to "default_execution").
      - parameters: The code that was executed.
      - path_executed: The working directory in which the code was run.
      - response: The output produced by the code or an error message if execution fails.
    
    Args:
        code: The Python code to execute.
    Returns:
        A JSON string containing execution details.
    """
    # Set a default execution name since no parameter is provided.
    execution_name = "default_execution"
    logger.info(f"Execution Name: {execution_name}")
    logger.info(f"Parameters: {json.dumps({'code': code})}")
    current_path = os.getcwd()
    logger.info(f"Path Executed In: {current_path}")
    
    output_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(output_capture), contextlib.redirect_stderr(output_capture):
            local_vars = {}
            exec(code, {"__builtins__": __builtins__}, local_vars)
        execution_output = output_capture.getvalue().strip()
        if not execution_output:
            execution_output = "Code executed successfully with no output."
        logger.info(f"Response: {execution_output}")
    except Exception as e:
        execution_output = f"Error executing code: {e}"
        logger.error(f"Response: {execution_output}")
    
    # Build a structured JSON response for better log analysis.
    response_details = {
        "execution_name": execution_name,
        "parameters": {"code": code},
        "path_executed": current_path,
        "response": execution_output,
    }
    return json.dumps(response_details, indent=2)

# ---------------------------------------------------------------------
# Tool for Output Validation Workflow
# ---------------------------------------------------------------------

def output_validation_guard(output: str) -> bool:
    """
    Checks if the output contains any incomplete or placeholder sections.
    
    Args:
        output: The text output to validate.
    Returns:
        True if valid; False if it contains any incomplete segments.
    """
    placeholders = ["todo", "mock code", "incomplete", "placeholder"]
    lower_output = output.lower()
    for ph in placeholders:
        if ph in lower_output:
            return False
    return True

@function_tool
def validate_output(output: str) -> str:
    """
    Validates the output to ensure it does not contain incomplete sections.
    If invalid, returns a message indicating failure; otherwise, confirms the output is valid.

    Args:
        output: The text output produced by a workflow.
    Returns:
        A message indicating whether the output passed validation.
    """
    if not output_validation_guard(output):
        return ("Output Validation Failed: Detected incomplete sections. "
                "Please reattempt the task with complete and valid output.")
    return "Output validation passed. The output is complete and valid."

# ---------------------------------------------------------------------
# Agent Definition and Configuration
# ---------------------------------------------------------------------

agent_instructions = ("""
As coding_agent, perform tasks such as file operations, directory listings, code execution, and managing coding workflows while adhering to the following guidelines.

- **File Operations**: Ensure all file write operations are confined to the launch directory. Every file write action must be logged accurately and in detail.
- **Output Validation**: Check your outputs thoroughly to make sure they contain no incomplete code segments. Validate all files for completeness before finalizing any operation.

# Steps

1. **File Operations**: 
   - Perform read, write, and modify operations only in the launch directory.
   - Log each file write operation with a timestamp, file path, and operation details.

2. **Directory Listings**: 
   - Retrieve a list of files and directories within the launch directory.
   - Ensure the inclusion of file attributes such as size and creation date if prompted.

3. **Code Execution**:
   - Execute code segments safely and efficiently.
   - Capture and log outputs including errors and run-time messages.

4. **Code Validation**:
   - After generating or modifying code, ensure all code blocks are complete and functional.
   - Validate syntax and logic before completing the task.

# Output Format

Produce outputs as clear, well-structured logs including necessary details such as timestamps, file paths, and validation results. Ensure the logging format is consistent and comprehensive.

# Examples

**Example 1**:
- **Task**: Write a file "example.txt" with content "Hello, World!" in the launch directory.
- **Output**: Log entry with timestamp indicating successful write operation and file path.

**Example 2**:
- **Task**: List all files in the launch directory.
- **Output**: A structured list of files with attributes such as name, size, and last modified date.

# Notes

- Maintain a consistent naming convention for logs and file paths.
- Implement error handling for any file operation failures and log them accordingly.
- Ensure code segments are syntactically correct and complete before execution.


""")

def create_coding_agent(model: str = "o3-mini") -> Agent:
    """
    Creates and returns a coding_agent instance using the specified model.
    
    Args:
        model: The model name to be used by the agent.
    Returns:
        An instance of the coding_agent.
    """
    return Agent(
        name="coding_agent",
        instructions=agent_instructions,
        model=model,
        tools=[read_file, write_file, list_directory, validate_output, execute_code],
    )

async def run_with_prompt(prompt: str, model: str):
    """
    Runs the coding_agent with a custom programming task prompt using the specified model.
    
    Args:
        prompt: A string representing the programming task (e.g., "create calc.py").
        model: The model name to use for this run.
    """
    agent = create_coding_agent(model)
    result = await Runner.run(agent, input=prompt)
    print("Custom Prompt Result:")
    print(result.final_output)
    print("-" * 60)

async def demo_workflows(model: str):
    """
    Runs demo workflows using a default prompt to showcase the coding_agent capabilities.
    
    Args:
        model: The model name to use for the demo workflows.
    """
    print(f"Running demo workflows with model: {model}")
    sample_prompt = "Create a file named demo.txt with content 'Hello, World!'"
    await run_with_prompt(sample_prompt, model)

# ---------------------------------------------------------------------
# Main Execution: Argument Parsing for Custom Prompt and Model Selection
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coding Agent Task Runner")
    parser.add_argument("--prompt", type=str, help="Programming task prompt for the coding agent")
    parser.add_argument("--model", type=str, default="o3-mini", help="Specify the model to use for the coding agent")
    args = parser.parse_args()

    import asyncio
    if args.prompt:
        asyncio.run(run_with_prompt(args.prompt, args.model))
    else:
        asyncio.run(demo_workflows(args.model))
