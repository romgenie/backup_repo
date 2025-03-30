#!/usr/bin/env python3
"""
Comprehensive DSPy Documentation Generator for AI-Scientist

This script:
1. Extracts general DSPy documentation from stanfordnlp_dspy.md
2. Creates AI-Scientist specific integration documentation
3. Organizes everything into a coherent documentation structure
"""

import os
import re
import argparse
from pathlib import Path

def create_directory(base_path, path):
    """Create directory if it doesn't exist"""
    full_path = os.path.join(base_path, path)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(f"Created directory: {full_path}")
    return full_path

def create_file(path, content):
    """Create a file with the specified content"""
    with open(path, 'w') as f:
        f.write(content)
    print(f"Created file: {path}")

def extract_general_docs(source_file, docs_dir):
    """Extract general DSPy documentation from stanfordnlp_dspy.md"""
    
    print(f"Extracting general DSPy documentation from {source_file}")
    
    # Create base directory structure
    core_dir = create_directory(docs_dir, "core")
    modules_dir = create_directory(docs_dir, "modules")
    optimizers_dir = create_directory(docs_dir, "optimizers")
    examples_dir = create_directory(docs_dir, "examples")
    usage_dir = create_directory(docs_dir, "usage")
    
    # Create overview/index files
    create_file(os.path.join(docs_dir, "README.md"), 
                "# DSPy Documentation\n\n"
                "This directory contains organized documentation for DSPy, tailored for the AI-Scientist project.\n\n"
                "## Contents\n\n"
                "- **core/**: Core components of DSPy (Signatures, Modules, Examples, Config)\n"
                "- **modules/**: Available DSPy modules (ChainOfThought, ProgramOfThought, ReAct, etc.)\n"
                "- **optimizers/**: Optimization techniques (BootstrapFewShot, MIPRO, etc.)\n"
                "- **examples/**: Common usage examples\n"
                "- **usage/**: General usage patterns and FAQs\n"
                "- **integration/**: AI-Scientist specific integration guides\n\n"
                "See the [AI-Scientist Guide](./AI-SCIENTIST-README.md) for project-specific integration details.\n")
    
    # Try to read the source file
    try:
        with open(source_file, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading source file: {e}")
        return
    
    # Extract documentation sections
    # This is a simplified approach - for a complex file, you might need more sophisticated parsing
    
    # Extract FAQs section
    faqs_match = re.search(r'## File: faqs\.md(.+?)## End of faqs\.md', content, re.DOTALL)
    if faqs_match:
        create_file(os.path.join(usage_dir, "faqs.md"), faqs_match.group(1).strip())
    
    # Extract Cheatsheet section (contains many code examples)
    cheatsheet_match = re.search(r'## File: cheatsheet\.md(.+?)## End of cheatsheet\.md', content, re.DOTALL)
    if cheatsheet_match:
        cheatsheet_content = cheatsheet_match.group(1).strip()
        
        # Process the cheatsheet content to extract different sections
        
        # Extract DataLoader examples
        dataloader_match = re.search(r'## DSPy DataLoaders(.+?)##', cheatsheet_content, re.DOTALL)
        if dataloader_match:
            create_file(os.path.join(usage_dir, "dataloaders.md"), 
                        "# DSPy DataLoaders\n\n" + dataloader_match.group(1).strip())
        
        # Extract Modules examples
        modules_match = re.search(r'## DSPy Programs(.+?)##', cheatsheet_content, re.DOTALL)
        if modules_match:
            create_file(os.path.join(examples_dir, "modules.md"), 
                        "# DSPy Modules Examples\n\n" + modules_match.group(1).strip())
        
        # Extract Metrics examples
        metrics_match = re.search(r'## DSPy Metrics(.+?)##', cheatsheet_content, re.DOTALL)
        if metrics_match:
            create_file(os.path.join(usage_dir, "metrics.md"), 
                        "# DSPy Metrics\n\n" + metrics_match.group(1).strip())
        
        # Extract Evaluation examples
        eval_match = re.search(r'## DSPy Evaluation(.+?)##', cheatsheet_content, re.DOTALL)
        if eval_match:
            create_file(os.path.join(usage_dir, "evaluation.md"), 
                        "# DSPy Evaluation\n\n" + eval_match.group(1).strip())
        
        # Extract Optimizer examples
        optimizers_match = re.search(r'## DSPy Optimizers(.+?)##', cheatsheet_content, re.DOTALL)
        if optimizers_match:
            create_file(os.path.join(optimizers_dir, "overview.md"), 
                        "# DSPy Optimizers\n\n" + optimizers_match.group(1).strip())
            
        # Extract Refine and BestofN examples
        refine_match = re.search(r'## DSPy `Refine` and `BestofN`(.+?)## End of cheatsheet\.md', cheatsheet_content, re.DOTALL)
        if refine_match:
            create_file(os.path.join(modules_dir, "refine_bestofn.md"), 
                        "# DSPy Refine and BestofN\n\n" + refine_match.group(1).strip())
    
    # Extract index/landing page content
    index_match = re.search(r'## File: index\.md(.+?)## End of index\.md', content, re.DOTALL)
    if index_match:
        create_file(os.path.join(docs_dir, "index.md"), 
                   "# DSPy: Programming—not prompting—LMs\n\n" + index_match.group(1).strip())
    
    # Extract roadmap content (for understanding upcoming features)
    roadmap_match = re.search(r'## File: roadmap\.md(.+?)## End of roadmap\.md', content, re.DOTALL)
    if roadmap_match:
        create_file(os.path.join(docs_dir, "roadmap.md"), 
                   "# DSPy Roadmap\n\n" + roadmap_match.group(1).strip())
    
    # Extract basic usage examples from the content
    # Create a structured overview document
    create_file(os.path.join(examples_dir, "README.md"),
                "# DSPy Usage Examples\n\n"
                "This directory contains examples extracted from the DSPy documentation.\n\n"
                "## Basic Examples\n\n"
                "```python\n"
                "import dspy\n\n"
                "# Configure language model\n"
                "lm = dspy.LM('openai/gpt-4o-mini')\n"
                "dspy.configure(lm=lm)\n\n"
                "# Define a simple module\n"
                "math = dspy.ChainOfThought(\"question -> answer: float\")\n"
                "result = math(question=\"What is the square root of 16?\")\n"
                "print(f\"Answer: {result.answer}\")\n"
                "```\n\n"
                "## RAG Examples\n\n"
                "```python\n"
                "import dspy\n\n"
                "def search_wikipedia(query: str) -> list[str]:\n"
                "    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)\n"
                "    return [x['text'] for x in results]\n\n"
                "rag = dspy.ChainOfThought('context, question -> response')\n\n"
                "question = \"What's the name of the castle that David Gregory inherited?\"\n"
                "rag(context=search_wikipedia(question), question=question)\n"
                "```\n")
    
    # Create a README for core components
    create_file(os.path.join(core_dir, "README.md"),
                "# DSPy Core Components\n\n"
                "This directory contains documentation for the core components of DSPy.\n\n"
                "## Core Components\n\n"
                "- **Signatures**: Define input/output specifications for modules\n"
                "- **Modules**: Base classes for building LM programs\n"
                "- **Examples**: Data containers for inputs/outputs\n"
                "- **Config**: Configuration settings for DSPy\n")
    
    # Create a signatures document
    create_file(os.path.join(core_dir, "signatures.md"),
                "# DSPy Signatures\n\n"
                "Signatures define the input and output structure for DSPy modules.\n\n"
                "## Basic Signature\n\n"
                "```python\n"
                "class BasicQA(dspy.Signature):\n"
                "    \"\"\"Answer questions with short factoid answers.\"\"\"\n\n"
                "    question = dspy.InputField()\n"
                "    answer = dspy.OutputField(desc=\"often between 1 and 5 words\")\n"
                "```\n\n"
                "## Short Form Signature\n\n"
                "You can also use a short-form string syntax:\n\n"
                "```python\n"
                "math = dspy.ChainOfThought(\"question -> answer: float\")\n"
                "```\n")
    
    print(f"General DSPy documentation extraction complete. Output in: {docs_dir}")

def add_ai_scientist_specific_docs(docs_dir):
    """Add AI-Scientist specific documentation files"""
    
    print(f"Adding AI-Scientist specific DSPy documentation to {docs_dir}")
    
    # Create integration directory
    integration_dir = create_directory(docs_dir, "integration")
    
    # Ensure modules directory exists
    modules_dir = os.path.join(docs_dir, "modules")
    if not os.path.exists(modules_dir):
        modules_dir = create_directory(docs_dir, "modules")
    
    # Add LLM integration documentation
    create_file(os.path.join(integration_dir, "llm_integration.md"),
                "# Integrating DSPy with AI-Scientist LLM System\n\n"
                "This guide explains how to integrate DSPy with the existing LLM system in AI-Scientist.\n\n"
                "## Overview\n\n"
                "The AI-Scientist project uses a custom LLM client in `ai_scientist/llm.py` that supports multiple LLM providers (OpenAI, Anthropic, etc.). "
                "DSPy has its own LLM client system that can be integrated with this existing system.\n\n"
                "## Integration Approach\n\n"
                "### Option 1: Using DSPy's LLM Client\n\n"
                "```python\n"
                "import dspy\n\n"
                "# Configure DSPy to use the same LLM as AI-Scientist\n"
                "# For OpenAI models\n"
                "lm = dspy.LM('openai/gpt-4o-mini')\n"
                "# For Anthropic models\n"
                "lm = dspy.LM('anthropic/claude-3-5-sonnet-20240620')\n"
                "# Configure DSPy to use this LLM\n"
                "dspy.configure(lm=lm)\n"
                "```\n\n"
                "### Option 2: Creating a Custom LLM Adapter\n\n"
                "You can create a custom adapter that wraps the AI-Scientist LLM client:\n\n"
                "```python\n"
                "import dspy\n"
                "from ai_scientist.llm import get_response_from_llm, create_client\n\n"
                "class AIScientistLM(dspy.LM):\n"
                "    def __init__(self, model_name):\n"
                "        self.client, self.model = create_client(model_name)\n"
                "        self.system_message = \"You are a helpful AI assistant.\"\n"
                "        \n"
                "    def basic_request(self, prompt, **kwargs):\n"
                "        response, _ = get_response_from_llm(\n"
                "            msg=prompt,\n"
                "            client=self.client,\n"
                "            model=self.model,\n"
                "            system_message=self.system_message,\n"
                "            temperature=kwargs.get('temperature', 0.7)\n"
                "        )\n"
                "        return [response]\n\n"
                "# Use the custom LLM adapter\n"
                "lm = AIScientistLM(\"gpt-4o-mini-2024-07-18\")\n"
                "dspy.configure(lm=lm)\n"
                "```\n")
    
    # Add experiment integration documentation
    create_file(os.path.join(integration_dir, "experiment_integration.md"),
                "# Integrating DSPy with AI-Scientist Experiments\n\n"
                "This guide provides examples of how to integrate DSPy with AI-Scientist experiments.\n\n"
                "## NanoGPT Experiments\n\n"
                "NanoGPT experiments can benefit from DSPy's ability to optimize prompts and create more structured LLM interactions.\n\n"
                "### Example: Enhancing Experiment Generation\n\n"
                "```python\n"
                "import dspy\n\n"
                "# Define a signature for experiment generation\n"
                "class ExperimentGenerator(dspy.Signature):\n"
                "    \"\"\"Generate experiments for a research idea.\"\"\"\n"
                "    \n"
                "    idea = dspy.InputField(desc=\"The research idea to explore\")\n"
                "    context = dspy.InputField(desc=\"Background information and constraints\")\n"
                "    experiments = dspy.OutputField(desc=\"List of experiments to run\")\n\n"
                "# Create a module using Chain of Thought reasoning\n"
                "generator = dspy.ChainOfThought(ExperimentGenerator)\n\n"
                "# Integrate with AI-Scientist perform_experiments function\n"
                "def enhanced_experiment_generation(idea_json):\n"
                "    # Extract info from idea\n"
                "    idea_desc = idea_json['description']\n"
                "    context = f\"Domain: {idea_json['domain']}\\nConstraints: GPU memory limit of 16GB\"\n"
                "    \n"
                "    # Generate experiments using DSPy\n"
                "    result = generator(idea=idea_desc, context=context)\n"
                "    \n"
                "    # Convert to the format expected by perform_experiments\n"
                "    # ...\n"
                "    \n"
                "    return experiments\n"
                "```\n\n"
                "## Optimizing the Experimental Process\n\n"
                "You can use DSPy optimizers to improve the quality of experiment generation:\n\n"
                "```python\n"
                "from dspy.teleprompt import BootstrapFewShot\n\n"
                "# Define a metric for evaluating experiment quality\n"
                "def experiment_quality_metric(example, pred):\n"
                "    # Implement your quality assessment logic\n"
                "    # Return a score between 0 and 1\n"
                "    return score\n\n"
                "# Create training examples\n"
                "trainset = [\n"
                "    dspy.Example(\n"
                "        idea=\"Improve language model training efficiency\",\n"
                "        context=\"Domain: NLP\\nConstraints: GPU memory limit of 16GB\",\n"
                "        experiments=\"[...]\"\n"
                "    ),\n"
                "    # Add more examples\n"
                "]\n\n"
                "# Optimize the generator\n"
                "optimizer = BootstrapFewShot(metric=experiment_quality_metric)\n"
                "optimized_generator = optimizer.compile(generator, trainset=trainset)\n"
                "```\n")
    
    # Add configuration integration documentation
    create_file(os.path.join(integration_dir, "config_integration.md"),
                "# Integrating DSPy with AI-Scientist Configuration\n\n"
                "This guide explains how to integrate DSPy with the AI-Scientist configuration system.\n\n"
                "## Overview\n\n"
                "AI-Scientist uses a configuration system based on the `configparser` module, loading from INI files."
                "DSPy has its own configuration system through environment variables and the `dspy.settings` object.\n\n"
                "## Integration Approach\n\n"
                "### Adding DSPy Configuration to AI-Scientist\n\n"
                "You can extend the existing `config.py` to include DSPy-specific settings:\n\n"
                "```python\n"
                "import configparser\n"
                "import os\n"
                "import dspy\n\n"
                "def load_config():\n"
                "    \"\"\"Load configuration from files and set up DSPy\"\"\"\n"
                "    config = configparser.ConfigParser()\n"
                "    base_dir = os.path.dirname(os.path.dirname(__file__))\n"
                "    config_rounds_path = os.path.join(base_dir, 'config_rounds.ini')\n"
                "    config_gpu_path = os.path.join(base_dir, 'config_gpu.ini')\n"
                "    config_dspy_path = os.path.join(base_dir, 'config_dspy.ini')\n"
                "    \n"
                "    # Load existing configs\n"
                "    if os.path.exists(config_rounds_path):\n"
                "        config.read(config_rounds_path)\n"
                "    if os.path.exists(config_gpu_path):\n"
                "        config.read(config_gpu_path)\n"
                "        \n"
                "    # Set up DSPy configuration\n"
                "    if os.path.exists(config_dspy_path):\n"
                "        config.read(config_dspy_path)\n"
                "        \n"
                "        # Configure DSPy caching\n"
                "        if 'cache' in config['dspy']:\n"
                "            cache_dir = config['dspy']['cache']\n"
                "            os.environ['DSPY_CACHE_DIR'] = cache_dir\n"
                "            \n"
                "        # Configure DSPy LM\n"
                "        if 'lm_provider' in config['dspy']:\n"
                "            provider = config['dspy']['lm_provider']\n"
                "            model = config['dspy']['lm_model']\n"
                "            lm = dspy.LM(f'{provider}/{model}')\n"
                "            dspy.configure(lm=lm)\n"
                "    \n"
                "    return config\n"
                "```\n\n"
                "### Example DSPy Configuration File (config_dspy.ini)\n\n"
                "```ini\n"
                "[dspy]\n"
                "cache = ./dspy_cache\n"
                "lm_provider = openai\n"
                "lm_model = gpt-4o-mini\n"
                "max_bootstrapped_demos = 3\n"
                "max_labeled_demos = 5\n"
                "```\n")
    
    # Create chain of thought documentation
    create_file(os.path.join(modules_dir, "chain_of_thought.md"),
                "# DSPy ChainOfThought Module\n\n"
                "The `ChainOfThought` module is one of DSPy's core reasoning modules. It prompts the language model to generate a step-by-step reasoning process before producing the final answer.\n\n"
                "## Basic Usage\n\n"
                "```python\n"
                "import dspy\n\n"
                "# Define a signature for your task\n"
                "class MathProblemSolver(dspy.Signature):\n"
                "    \"\"\"Solve math problems step by step.\"\"\"\n"
                "    \n"
                "    problem = dspy.InputField()\n"
                "    reasoning = dspy.OutputField(desc=\"Step-by-step reasoning process\")\n"
                "    solution = dspy.OutputField(desc=\"The final numerical answer\")\n\n"
                "# Create a ChainOfThought module with this signature\n"
                "solver = dspy.ChainOfThought(MathProblemSolver)\n\n"
                "# Use the module\n"
                "result = solver(problem=\"If 3x + 7 = 22, what is the value of x?\")\n"
                "print(f\"Reasoning: {result.reasoning}\")\n"
                "print(f\"Solution: {result.solution}\")\n"
                "```\n\n"
                "## AI-Scientist Integration\n\n"
                "ChainOfThought can be used to enhance various aspects of the AI-Scientist workflow:\n\n"
                "1. **Idea Generation**: Use it to create more structured and well-reasoned research ideas\n"
                "2. **Experiment Design**: Generate experimental setups with clear reasoning about methodological choices\n"
                "3. **Result Analysis**: Analyze experimental results with step-by-step reasoning\n\n"
                "### Example: Enhanced Review Process\n\n"
                "```python\n"
                "import dspy\n"
                "from ai_scientist.perform_review import perform_review\n\n"
                "class ExperimentReviewer(dspy.Signature):\n"
                "    \"\"\"Review AI experiment results thoroughly.\"\"\"\n"
                "    \n"
                "    experiment_results = dspy.InputField()\n"
                "    evaluation_criteria = dspy.InputField()\n"
                "    reasoning = dspy.OutputField(desc=\"Detailed analysis of results\")\n"
                "    review = dspy.OutputField(desc=\"Final review summary and recommendations\")\n\n"
                "reviewer = dspy.ChainOfThought(ExperimentReviewer)\n"
                "```\n")
    
    # Create react documentation
    create_file(os.path.join(modules_dir, "react.md"),
                "# DSPy ReAct Module\n\n"
                "The `ReAct` module implements the Reasoning and Acting framework, enabling LLMs to interleave reasoning steps with actions.\n\n"
                "## Basic Usage\n\n"
                "```python\n"
                "import dspy\n\n"
                "# Define a signature for your task\n"
                "class ProblemSolver(dspy.Signature):\n"
                "    \"\"\"Solve complex problems that require external tools.\"\"\"\n"
                "    \n"
                "    problem = dspy.InputField()\n"
                "    solution = dspy.OutputField()\n\n"
                "# Create a ReAct module\n"
                "solver = dspy.ReAct(ProblemSolver)\n\n"
                "# Use the module\n"
                "result = solver(problem=\"What is the population of the capital of France?\")\n"
                "print(f\"Solution: {result.solution}\")\n"
                "```\n\n"
                "## AI-Scientist Integration\n\n"
                "ReAct is particularly useful for AI-Scientist tasks that require interacting with external tools or APIs:\n\n"
                "1. **Data Collection**: Gathering research papers or datasets needed for experiments\n"
                "2. **Experiment Execution**: Running experiments with external tools and analyzing results\n"
                "3. **Literature Review**: Searching and analyzing related work\n\n"
                "### Example: Enhanced Experiment Runner\n\n"
                "```python\n"
                "import dspy\n"
                "import torch\n"
                "import numpy as np\n\n"
                "# Define available tools\n"
                "def run_nanoGPT_experiment(params):\n"
                "    # Run NanoGPT with given parameters\n"
                "    # Return results\n"
                "    return results\n\n"
                "def analyze_results(data):\n"
                "    # Analyze experimental results\n"
                "    return analysis\n\n"
                "# Define a signature for experiment running\n"
                "class ExperimentRunner(dspy.Signature):\n"
                "    \"\"\"Run and analyze NanoGPT experiments.\"\"\"\n"
                "    \n"
                "    experiment_config = dspy.InputField()\n"
                "    results = dspy.OutputField()\n\n"
                "# Create a ReAct module with this signature\n"
                "runner = dspy.ReAct(ExperimentRunner)\n"
                "# Define the available tools\n"
                "runner.set_tools([\n"
                "    run_nanoGPT_experiment,\n"
                "    analyze_results\n"
                "])\n\n"
                "# Use the module\n"
                "result = runner(experiment_config={\"model_size\": \"124M\", \"learning_rate\": 0.0003})\n"
                "```\n")
    
    # Create program of thought documentation
    create_file(os.path.join(modules_dir, "program_of_thought.md"),
                "# DSPy ProgramOfThought Module\n\n"
                "The `ProgramOfThought` module extends chain-of-thought reasoning with more structured programmatic elements.\n\n"
                "## Basic Usage\n\n"
                "```python\n"
                "import dspy\n\n"
                "# Define a signature for your task\n"
                "class MathProblemSolver(dspy.Signature):\n"
                "    \"\"\"Solve math problems using a programmatic approach.\"\"\"\n"
                "    \n"
                "    problem = dspy.InputField()\n"
                "    solution = dspy.OutputField()\n\n"
                "# Create a ProgramOfThought module\n"
                "solver = dspy.ProgramOfThought(MathProblemSolver)\n\n"
                "# Use the module\n"
                "result = solver(problem=\"If 3x + 7 = 22, what is the value of x?\")\n"
                "print(f\"Solution: {result.solution}\")\n"
                "```\n\n"
                "## AI-Scientist Integration\n\n"
                "ProgramOfThought is well-suited for AI-Scientist tasks that require structured reasoning and programming concepts:\n\n"
                "1. **Experimental Design**: Creating structured experimental protocols\n"
                "2. **Algorithm Development**: Developing new algorithms or modifications to existing ones\n"
                "3. **Data Analysis**: Structured analysis of experimental results\n\n"
                "### Example: NanoGPT Modification Designer\n\n"
                "```python\n"
                "import dspy\n\n"
                "class NanoGPTModifier(dspy.Signature):\n"
                "    \"\"\"Design modifications to the NanoGPT architecture.\"\"\"\n"
                "    \n"
                "    objective = dspy.InputField(desc=\"What we want to improve about NanoGPT\")\n"
                "    constraints = dspy.InputField(desc=\"Computational and other constraints\")\n"
                "    design = dspy.OutputField(desc=\"Detailed design of the modification\")\n"
                "    implementation_steps = dspy.OutputField(desc=\"Steps to implement the modification\")\n\n"
                "# Create a ProgramOfThought module\n"
                "modifier = dspy.ProgramOfThought(NanoGPTModifier)\n\n"
                "# Use the module\n"
                "result = modifier(\n"
                "    objective=\"Improve training efficiency while maintaining model quality\",\n"
                "    constraints=\"Must run on a single 16GB GPU. Cannot increase model parameters.\"\n"
                ")\n"
                "```\n")
    
    # Create retrieve documentation
    create_file(os.path.join(modules_dir, "retrieve.md"),
                "# DSPy Retrieve Module\n\n"
                "The `Retrieve` module provides retrieval-augmented generation capabilities, allowing you to retrieve relevant information from a corpus.\n\n"
                "## Basic Usage\n\n"
                "```python\n"
                "import dspy\n\n"
                "# Set up a retrieval model (RM)\n"
                "retriever = dspy.ColBERTv2(url=\"http://example.com/colbert\")\n"
                "dspy.settings.configure(rm=retriever)\n\n"
                "# Create a Retrieve module\n"
                "retrieve = dspy.Retrieve(k=3)  # Retrieve top 3 passages\n\n"
                "# Use the module\n"
                "result = retrieve(\"What are the key components of a transformer architecture?\")\n"
                "for i, passage in enumerate(result.passages):\n"
                "    print(f\"Passage {i+1}: {passage}\")\n"
                "```\n\n"
                "## AI-Scientist Integration\n\n"
                "Retrieve is valuable for AI-Scientist tasks that require knowledge retrieval:\n\n"
                "1. **Literature Review**: Retrieving relevant papers and research findings\n"
                "2. **Background Knowledge**: Retrieving factual information for experiment design\n"
                "3. **Citation Generation**: Finding relevant citations for paper writing\n\n"
                "### Example: Enhanced Literature Review\n\n"
                "```python\n"
                "import dspy\n\n"
                "# Set up a retrieval model with research papers corpus\n"
                "papers_retriever = dspy.ColBERTv2(url=\"http://example.com/papers_corpus\")\n"
                "dspy.settings.configure(rm=papers_retriever)\n\n"
                "# Define a signature for literature review\n"
                "class LiteratureReviewer(dspy.Signature):\n"
                "    \"\"\"Review literature relevant to a research topic.\"\"\"\n"
                "    \n"
                "    topic = dspy.InputField()\n"
                "    context = dspy.InputField()\n"
                "    review = dspy.OutputField()\n\n"
                "# Create a module combining retrieval and reasoning\n"
                "class RAGReviewer(dspy.Module):\n"
                "    def __init__(self):\n"
                "        super().__init__()\n"
                "        self.retrieve = dspy.Retrieve(k=5)\n"
                "        self.summarize = dspy.ChainOfThought(LiteratureReviewer)\n"
                "    \n"
                "    def forward(self, topic):\n"
                "        # Retrieve relevant papers\n"
                "        retrieved = self.retrieve(topic)\n"
                "        # Summarize the findings\n"
                "        review = self.summarize(topic=topic, context=retrieved.passages)\n"
                "        return review\n\n"
                "# Use the module\n"
                "reviewer = RAGReviewer()\n"
                "result = reviewer(\"Transformer architecture efficiency improvements\")\n"
                "print(result.review)\n"
                "```\n")
    
    # Create an AI-Scientist specific README
    create_file(os.path.join(docs_dir, "AI-SCIENTIST-README.md"),
                "# DSPy Integration with AI-Scientist\n\n"
                "This documentation provides guidance on integrating DSPy with the AI-Scientist project.\n\n"
                "## What is DSPy?\n\n"
                "DSPy is a framework for programming language models through modular components with optimizable prompts. "
                "It allows you to build complex LLM pipelines using Python code rather than brittle prompts.\n\n"
                "## How DSPy Can Enhance AI-Scientist\n\n"
                "1. **Structured LLM Interactions**: Replace ad-hoc prompting with structured DSPy modules\n"
                "2. **Optimizable Pipelines**: Automatically improve prompts using DSPy optimizers\n"
                "3. **Modular Design**: Build complex reasoning pipelines with reusable components\n"
                "4. **Enhanced Retrieval**: Improve literature review and knowledge retrieval\n\n"
                "## Key Documentation Sections\n\n"
                "- **Integration**: Guides for integrating DSPy with AI-Scientist's existing systems\n"
                "  - [LLM Integration](./integration/llm_integration.md)\n"
                "  - [Experiment Integration](./integration/experiment_integration.md)\n"
                "  - [Configuration Integration](./integration/config_integration.md)\n"
                "- **Modules**: Documentation for key DSPy modules relevant to AI-Scientist\n"
                "  - [ChainOfThought](./modules/chain_of_thought.md)\n"
                "  - [ProgramOfThought](./modules/program_of_thought.md)\n"
                "  - [ReAct](./modules/react.md)\n"
                "  - [Retrieve](./modules/retrieve.md)\n"
                "- **Core Components**: Documentation for DSPy's core components\n"
                "  - [Signatures](./core/signatures.md)\n"
                "- **Usage Examples**: Practical examples of using DSPy in AI-Scientist contexts\n\n"
                "## Getting Started\n\n"
                "To get started with DSPy integration, first install DSPy:\n\n"
                "```bash\n"
                "pip install -U dspy\n"
                "```\n\n"
                "Then, set up a basic DSPy configuration in your AI-Scientist project:\n\n"
                "```python\n"
                "import dspy\n\n"
                "# Configure DSPy to use the same LLM as AI-Scientist\n"
                "lm = dspy.LM('openai/gpt-4o-mini')\n"
                "dspy.configure(lm=lm)\n"
                "```\n\n"
                "## Step-by-Step Integration Plan\n\n"
                "1. Start with a small, non-critical component of AI-Scientist\n"
                "2. Refactor it to use DSPy modules and signatures\n"
                "3. Test the refactored component thoroughly\n"
                "4. Gradually expand DSPy usage to other components\n"
                "5. Implement optimizers to improve prompt quality\n\n"
                "## Best Practices\n\n"
                "- Keep DSPy modules focused on single responsibilities\n"
                "- Use signatures to clearly define input/output interfaces\n"
                "- Leverage DSPy optimizers to improve performance\n"
                "- Start with ChainOfThought for reasoning tasks\n"
                "- Use ReAct for tasks requiring tool use or external interaction\n"
                "- Implement ProgramOfThought for complex, structured reasoning\n")
    
    print(f"AI-Scientist specific documentation added to {docs_dir}")

def generate_dspy_docs(source_file, dest_dir):
    """Generate comprehensive DSPy documentation for AI-Scientist project"""
    
    # Create the base directory
    docs_dir = create_directory(dest_dir, "dspy_docs")
    
    # Extract general DSPy documentation
    extract_general_docs(source_file, docs_dir)
    
    # Add AI-Scientist specific documentation
    add_ai_scientist_specific_docs(docs_dir)
    
    print(f"DSPy documentation generation complete. Documentation available at: {docs_dir}")
    print(f"Main entry point: {os.path.join(docs_dir, 'AI-SCIENTIST-README.md')}")

def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive DSPy documentation for AI-Scientist")
    parser.add_argument("--source", default="/Users/timgregg/mcp/stanfordnlp_dspy.md", 
                        help="Path to stanfordnlp_dspy.md file")
    parser.add_argument("--dest", default="/Users/timgregg/mcp/dspy_docs_ai_scientist", 
                        help="Destination directory for documentation")
    args = parser.parse_args()
    
    generate_dspy_docs(args.source, args.dest)

if __name__ == "__main__":
    main()