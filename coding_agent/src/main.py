"
Improved main.py

This module implements coding_agent, a modular coding assistant capable of performing file operations,
directory listings, code execution, and managing various coding workflows. Enhancements include:
- Use of pathlib for improved path handling.
- Consistent logging with clear structured messages.
- Enhanced docstrings with type hints for better readability.
- Minor refactorings for clearer structure and readability.

Author: Coding Agent
""

import os
import datetime
import logging
import json
import argparse
import io
import contextlib
from pathlib import Path
from agents import Agent, Runner, function_tool

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# The allowed base directory is the directory from which the agent is launched.
BASE_DIR = Path.cwd()


def is_within_base_directory(path: str) -> bool:
    """
    Check if the given path is within the BASE_DIR.

    Args:
        path (str): The target file path.
    Returns:
        bool: True if the absolute path is within BASE_DIR, else False.
    """
    try:
        abs_path = Path(path).resolve()
        return str(abs_path).startswith(str(BASE_DIR.resolve()))
    except Exception as e:
        logger.error(f"Error resolving path '{path}': {e}")
        return False

# ---------------------------------------------------------------------
# Tool Definitions: File and Directory Operations
# ---------------------------------------------------------------------

@function_tool

def read_file(path: str) -> str:
    """
    Reads the contents of a file given its path.

    Args:
        path (str): The path to the file.

    Returns:
        str: The file content or an error message if the file does not exist or cannot be read.
    """
    file_path = Path(path)
    if not file_path.exists():
        return f"Error: File '{path}' not found."
    try:
        return file_path.read_text()
    except Exception as e:
        return f"Error reading file '{path}': {e}"

@function_tool

def write_file(path: str, content: str) -> str:
    """
    Writes content to a file specified by its path.
    Guardrail: Only allows writing to files within BASE_DIR.
    Tracing: Logs every file write operation with details.
    Enhancement: Automatically creates the target directory if it does not exist.

    Args:
        path (str): The target file path.
        content (str): The content to write.

    Returns:
        str: Success message or guardrail error message.
    """
    if not is_within_base_directory(path):
        return f"Guardrail Triggered: Write operation to '{path}' is not allowed (outside {BASE_DIR})."

    try:
        target_path = Path(path).resolve()
        if not target_path.parent.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory created: '{target_path.parent}'")
        target_path.write_text(content)
        timestamp = datetime.datetime.now().isoformat()
        logger.info(f"File written: '{target_path}' at {timestamp} (wrote {len(content)} characters).")
        return f"File '{target_path.name}' has been written successfully."
    except Exception as e:
        return f"Error writing file '{path}': {e}"

@function_tool

def list_directory(path: str) -> list:
    """
    Lists the contents of a directory. If an empty string is provided, BASE_DIR is used.

    Args:
        path (str): The directory path (or empty for BASE_DIR).

    Returns:
        list: List of directory contents or error message(s) if invalid.
    """
    target_dir = Path(path) if path else BASE_DIR
    if not target_dir.is_dir():
        return [f"Error: '{path}' is not a valid directory."]
    try:
        return [item.name for item in target_dir.iterdir()]
    except Exception as e:
        return [f"Error listing directory '{path}': {e}"]

# ---------------------------------------------------------------------
# Tool for Code Execution
# ---------------------------------------------------------------------

@function_tool

def execute_code(code: str) -> str:
    """
    Executes the given Python code and returns a structured JSON containing execution details.

    The returned JSON includes:
      - execution_name: A fixed name ("default_execution").
      - parameters: The code that was executed.
      - path_executed: The working directory in which the code was run.
      - response: The output produced by the code or error message if it fails.

    Args:
        code (str): The Python code to execute.

    Returns:
        str: A JSON string containing execution details.
    """
    execution_name = "default_execution"
    logger.info(f"Execution Name: {execution_name}")
    logger.info(f"Parameters: {json.dumps({'code': code})}")
    current_path = str(Path.cwd())
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
    Validates that the output does not contain incomplete or placeholder segments.

    Args:
        output (str): The text output to validate.

    Returns:
        bool: True if output is valid, False otherwise.
    """
    placeholders = ["todo", "mock code", "incomplete", "placeholder"]
    return not any(ph in output.lower() for ph in placeholders)

@function_tool

def validate_output(output: str) -> str:
    """
    Validates the output to ensure it contains no incomplete sections.

    Args:
        output (str): The text output produced by a workflow.

    Returns:
        str: Validation message indicating whether the output is acceptable.
    """
    if not output_validation_guard(output):
        return ("Output Validation Failed: Detected incomplete sections. "
                "Please reattempt the task with complete and valid output.")
    return "Output validation passed. The output is complete and valid."

# ---------------------------------------------------------------------
# Agent Definition and Configuration
# ---------------------------------------------------------------------

agent_instructions = (
    "You are coding_agent, a modular coding assistant capable of performing file operations, "
    "directory listings, code execution, and handling various coding workflows. "
    "Ensure file write operations occur within the launch directory and are properly logged. "
    "Validate outputs to ensure no incomplete code segments exist."
)

def create_coding_agent(model: str = "o3-mini") -> Agent:
    """
    Creates and returns an instance of coding_agent with the specified model.

    Args:
        model (str): The model name for the agent. Default is "o3-mini".

    Returns:
        Agent: An instance of the coding_agent.
    """
    return Agent(
        name="coding_agent",
        instructions=agent_instructions,
        model=model,
        tools=[read_file, write_file, list_directory, validate_output, execute_code],
    )

async def run_with_prompt(prompt: str, model: str) -> None:
    """
    Runs the coding_agent with a custom programming task prompt using the specified model.

    Args:
        prompt (str): The programming task prompt (e.g., "create calc.py").
        model (str): The model name to use for the coding_agent.
    """
    agent = create_coding_agent(model)
    result = await Runner.run(agent, input=prompt)
    print("Custom Prompt Result:")
    print(result.final_output)
    print("-" * 60)

async def demo_workflows(model: str) -> None:
    """
    Demonstrates the capabilities of coding_agent through sample workflows.

    Args:
        model (str): The model name to use for demo workflows.
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
    try:
        if args.prompt:
            asyncio.run(run_with_prompt(args.prompt, args.model))
        else:
            asyncio.run(demo_workflows(args.model))
    except Exception as main_err:
        logger.error(f"Error in main execution: {main_err}")
