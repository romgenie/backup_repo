# Archive Folder

This folder contains files and directories that were created during the refactoring process but are not currently being used in the application. They're kept here for reference in case they're needed in future development.

## Contents

### Services

The `services` folder was intended to provide business logic services separate from the UI and application layers. The refactoring ultimately used a different approach, but these files contain potentially useful code.

- `env_service.py` - Environment variable handling functions
- `mcp_service.py` - MCP server interaction services

These services could be integrated in the future for better separation of concerns.