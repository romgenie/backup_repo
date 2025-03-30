#!/bin/bash

# Function to run a command with retry logic.
run_with_retry() {
  local max_attempts=3
  local delay=5
  local attempt=1

  until "$@"; do
    if [ $attempt -ge $max_attempts ]; then
      echo "Command '$*' failed after $attempt attempts."
      return 1
    fi
    echo "Command '$*' failed. Retrying in $delay seconds... (Attempt: $((attempt+1)) of $max_attempts)"
    sleep "$delay"
    attempt=$((attempt + 1))
  done
}

# Check if a company argument is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <company>"
  exit 1
fi

COMPANY="$1"

# Check if gh CLI is authenticated
if ! gh auth status > /dev/null 2>&1; then
  echo "Error: gh CLI is not authenticated. Please run 'gh auth login' and try again."
  exit 1
fi

# Create a directory for the company
mkdir -p "$COMPANY"

# File to track repositories that have been cloned already.
CLONED_REPOS_FILE="$COMPANY/.cloned_repos"
touch "$CLONED_REPOS_FILE"

# Get repository names using JSON output for robust parsing.
REPO_NAMES=$(gh repo list "$COMPANY" --limit 1000 --json name -q '.[].name')

# Check if the output is empty or malformed
if [ -z "$REPO_NAMES" ]; then
  echo "No repositories found for '$COMPANY' or the output is empty. Please check the company name or your access rights."
  exit 0
fi

# Set IFS to newline to handle spaces and special characters properly
OLDIFS=$IFS
IFS=$'\n'

# Loop through each repository name
for REPO in $REPO_NAMES; do
  REPO_DIR="$COMPANY/$REPO"
  
  # Check if this repo has been cloned previously.
  if grep -Fxq "$REPO" "$CLONED_REPOS_FILE"; then
    echo "Repository '$REPO' was previously cloned. Skipping clone."
    # If the directory exists, you might still want to check for updates.
    if [ -d "$REPO_DIR" ]; then
      echo "Repository '$REPO' directory exists. Checking for updates..."
      cd "$REPO_DIR" || { echo "Error: Unable to enter directory '$REPO_DIR'"; continue; }
      if [ -n "$(git status --porcelain)" ]; then
        echo "Warning: '$REPO' has local modifications. Skipping pull to avoid merge conflicts."
      else
        echo "Pulling latest changes for '$REPO'..."
        run_with_retry git pull || echo "Error pulling repository '$REPO'. Please check the repository status."
      fi
      cd - > /dev/null
    fi
    continue
  fi

  # If the repository directory exists, update it.
  if [ -d "$REPO_DIR" ]; then
    if [ -d "$REPO_DIR/.git" ]; then
      echo "Repository '$REPO' already exists."
      cd "$REPO_DIR" || { echo "Error: Unable to enter directory '$REPO_DIR'"; continue; }
      if [ -n "$(git status --porcelain)" ]; then
        echo "Warning: '$REPO' has local modifications. Skipping pull to avoid merge conflicts."
      else
        echo "Pulling latest changes for '$REPO'..."
        run_with_retry git pull || echo "Error pulling repository '$REPO'. Please check the repository status."
      fi
      cd - > /dev/null
      # Record that the repository was cloned.
      if ! grep -Fxq "$REPO" "$CLONED_REPOS_FILE"; then
        echo "$REPO" >> "$CLONED_REPOS_FILE"
      fi
    else
      echo "Warning: Directory '$REPO_DIR' exists but is not a valid Git repository. Skipping."
    fi
  else
    # Clone the repository if it hasn't been cloned before.
    echo "Cloning repository '$REPO'..."
    run_with_retry gh repo clone "$COMPANY/$REPO" "$REPO_DIR" || { 
      echo "Error cloning repository '$REPO'. Please check your permissions or rate limits."; 
      continue; 
    }
    echo "$REPO" >> "$CLONED_REPOS_FILE"
  fi

  # Optional: Add a small delay between processing repositories to mitigate rate limiting.
  sleep 1
done

# Reset IFS
IFS=$OLDIFS
