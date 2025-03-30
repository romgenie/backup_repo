# Personal Assistant Voice Application

A modular voice assistant application that can answer queries about accounts, product information, and real-time web searches.

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure your environment variables in the `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   VECTOR_STORE_ID=your_vector_store_id_here
   ```

## Running the Application

You can run the application in several ways:

### Interactive Menu
```bash
python -m src.main
```

### Command Line Mode
```bash
# Initialize vector store
python -m src.main setup

# Display vector store information
python -m src.main info

# Text mode
python -m src.main text

# Voice mode with default profile
python -m src.main voice

# Voice mode with specific profile
python -m src.main voice upbeat
python -m src.main voice character
```

## Project Structure

The project follows a modular structure with dedicated directories for agents, tools, and utilities:

```
src/
├── agents/           # Each agent has its own directory
│   ├── account/      # Account agent
│   ├── knowledge/    # Product knowledge agent
│   ├── search/       # Web search agent
│   └── triage/       # Main routing agent
├── tools/            # Each tool type has its own directory
│   ├── account/      # Account tools
│   ├── web_search/   # Web search tools
│   └── file_search/  # File search tools
├── voice_pipeline/   # Voice interaction components
├── utils/            # Utility functions
├── config/           # Configuration settings
└── data/             # Data resources
    ├── audio/        # Audio files
    └── knowledge/    # Knowledge files
```

See `plans.md` for detailed structure information.

## Environment Variables

The application uses the following environment variables from the `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key
- `VECTOR_STORE_ID`: ID of your OpenAI vector store
- `SAMPLE_RATE`: Audio sample rate (default: 16000)
- `DEFAULT_MODE`: Default application mode (text or voice)
- `DEFAULT_VOICE_PROFILE`: Default voice profile (default, upbeat, character)

## Voice Profiles

The application supports three voice profiles:
- `default`: Clear, professional, and informative tone
- `upbeat`: Friendly, warm, and supportive tone
- `character`: Dramatic, noble, heroic tone with an archaic quality

## Vector Store Management

The application includes a built-in vector store management system that automatically sets up a vector store for your knowledge base.

### Automatic Setup

The application will check for a valid vector store ID but will NOT automatically create one.
Instead, a warning will be shown if no vector store is configured.

To explicitly create a vector store, run the setup command:

```bash
python -m src.main setup
```

This will:
1. Create a new vector store (even if one already exists)
2. Upload knowledge files from `src/data/knowledge/`
3. Save the vector store ID to your `.env` file

**IMPORTANT**: Creating a vector store will count against your OpenAI API usage. Only run the setup command when you intend to create a new vector store.

### Manual Setup Options

1. Use the OpenAI Platform website:
   - Go to [platform.openai.com/storage](https://platform.openai.com/storage)
   - Create a vector store and upload documents
   - Copy the vector store ID to your `.env` file

2. Use the Vector Store Agent interactively:
   ```python
   from agents import Runner, trace
   from src.agents import create_vector_store_agent
   
   # Create vector store agent
   vector_store_agent = create_vector_store_agent()
   
   # Create vector store
   with trace("Vector Store Manager"):
       result = await Runner.run(vector_store_agent, "Create a new vector store named 'ACME Shop Products'")
   
   # Upload knowledge files
   with trace("Vector Store Manager"):
       result = await Runner.run(vector_store_agent, "Upload file 'src/data/knowledge/acme_product_catalogue.pdf' to the vector store")
   ```

### Troubleshooting

If you see the error `Invalid type for 'tools[0].vector_store_ids[0]': expected a string, but got null instead`, it means:
1. No valid vector store ID is available
2. You need to run `python -m src.main setup` to create one