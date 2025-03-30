"""
Response types and reasoning model support for the SDK Playground.
This module provides components for configuring different response formats and reasoning models.
"""
import streamlit as st
import json
from typing import Dict, Any, Optional, List, Union


class ResponseType:
    """Definition of a response type with configuration options."""
    
    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        config_options: Dict[str, Dict] = None,
        example_output: str = None
    ):
        """
        Initialize a response type.
        
        Args:
            id: Unique identifier
            name: Display name
            description: Description of the response type
            config_options: Configuration options for this response type
            example_output: Example output text
        """
        self.id = id
        self.name = name
        self.description = description
        self.config_options = config_options or {}
        self.example_output = example_output
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API calls."""
        return {
            "id": self.id,
            "name": self.name,
            "config": {}  # Populated with UI values
        }


# Define common response types
RESPONSE_TYPES = [
    ResponseType(
        id="default",
        name="Default Response",
        description="Standard response without specific format requirements.",
        example_output="This is a standard text response from the model."
    ),
    ResponseType(
        id="json_mode",
        name="JSON Response",
        description="Response formatted as valid JSON. The model will produce syntactically valid JSON.",
        config_options={
            "format": {
                "type": "select",
                "options": ["object", "array"],
                "default": "object",
                "description": "The expected JSON format"
            }
        },
        example_output="""```json
{
  "analysis": {
    "summary": "This is a concise summary of the analysis.",
    "key_points": [
      "First key point with supporting details.",
      "Second key point with examples.",
      "Third key point with implications."
    ],
    "recommendation": "A clear recommendation based on the analysis."
  }
}
```"""
    ),
    ResponseType(
        id="reasoning",
        name="Chain-of-Thought Reasoning",
        description="Explicit step-by-step reasoning process to arrive at a conclusion.",
        config_options={
            "verbosity": {
                "type": "select",
                "options": ["concise", "detailed", "comprehensive"],
                "default": "detailed",
                "description": "Level of detail in the reasoning steps"
            },
            "format": {
                "type": "select",
                "options": ["numbered", "bulleted", "paragraph"],
                "default": "numbered",
                "description": "Format of the reasoning steps"
            }
        },
        example_output="""To solve this problem, I'll approach it step by step:

1. First, I'll identify the key variables in the problem.
2. Next, I'll establish the relationships between these variables.
3. Then, I'll apply the relevant formulas to solve for the unknown.
4. Finally, I'll verify my answer by checking if it satisfies all conditions.

Following this approach:
- Given information: A car travels 150 miles in 3 hours.
- Step 1: The key variable is the speed of the car.
- Step 2: Speed = Distance ÷ Time
- Step 3: Speed = 150 miles ÷ 3 hours = 50 miles per hour
- Step 4: Verification: At 50 mph for 3 hours, the car would travel 50 × 3 = 150 miles. ✓

Therefore, the car's speed is 50 miles per hour."""
    ),
    ResponseType(
        id="analytical",
        name="Analytical Assessment",
        description="Structured analysis with evidence-based evaluation and balanced perspectives.",
        config_options={
            "include_uncertainty": {
                "type": "checkbox",
                "default": True,
                "description": "Explicitly acknowledge areas of uncertainty"
            },
            "evaluation_criteria": {
                "type": "multiselect",
                "options": ["factual accuracy", "logical consistency", "practical implications", "ethical considerations"],
                "default": ["factual accuracy", "logical consistency"],
                "description": "Criteria to include in the analysis"
            }
        },
        example_output="""# Analysis of Climate Change Mitigation Strategies

## Summary
This analysis evaluates three primary climate change mitigation strategies based on effectiveness, feasibility, and cost.

## Evaluation

### Strategy 1: Renewable Energy Transition
- **Effectiveness**: High (70-80% emission reduction potential)
- **Feasibility**: Medium (requires significant infrastructure changes)
- **Cost**: High initial investment, lower long-term costs
- **Uncertainties**: Technology adoption rate, grid stability challenges

### Strategy 2: Carbon Capture and Sequestration
- **Effectiveness**: Medium (40-60% emission reduction potential)
- **Feasibility**: Medium-Low (technology still developing)
- **Cost**: High with unclear return on investment
- **Uncertainties**: Long-term storage stability, scalability

### Strategy 3: Policy and Behavior Change
- **Effectiveness**: Variable (20-70% depending on implementation)
- **Feasibility**: High (can be implemented immediately)
- **Cost**: Low direct costs, potential economic impacts
- **Uncertainties**: Public and political acceptance, enforcement

## Recommendation
A combined approach emphasizing renewable energy while supporting policy changes and targeted carbon capture represents the most balanced strategy, acknowledging that no single approach offers a complete solution."""
    ),
    ResponseType(
        id="instructional",
        name="Step-by-Step Instructions",
        description="Clear procedural instructions with sequential steps.",
        config_options={
            "detail_level": {
                "type": "select",
                "options": ["basic", "intermediate", "expert"],
                "default": "intermediate",
                "description": "Level of detail in the instructions"
            },
            "include_warnings": {
                "type": "checkbox",
                "default": True,
                "description": "Include cautions and warnings"
            }
        },
        example_output="""# How to Change a Flat Tire

## Tools You'll Need
- Jack
- Lug wrench
- Spare tire
- Wheel wedges

## Step-by-Step Instructions

### 1. Prepare the Vehicle
- Find a safe, flat location away from traffic
- Apply the parking brake
- Turn on hazard lights
- Place wheel wedges opposite the flat tire

⚠️ **WARNING**: Never change a tire on an incline or soft ground.

### 2. Access the Spare Tire
- Remove the spare tire and tools from your trunk or storage area
- Place them nearby for easy access

### 3. Loosen the Lug Nuts
- Use the lug wrench to loosen each lug nut by turning counterclockwise
- Only loosen them about ¼ turn - don't remove them yet

⚠️ **WARNING**: Lug nuts may be tight. Use body weight if necessary, but maintain proper balance.

### 4. Jack Up the Vehicle
- Position the jack under the designated jack point
- Raise the vehicle until the flat tire is about 6 inches off the ground

### 5. Remove Lug Nuts and Tire
- Completely remove the loosened lug nuts
- Carefully pull the flat tire straight toward you to remove it
- Set it aside

### 6. Mount the Spare Tire
- Align the spare tire with the wheel bolts
- Push it all the way onto the wheel mount

### 7. Replace and Tighten Lug Nuts
- Hand-tighten all lug nuts
- Tighten in a star pattern for even pressure
- Use the wrench to secure them, but don't fully tighten yet

### 8. Lower the Vehicle
- Carefully lower the vehicle to the ground
- Remove the jack

### 9. Fully Tighten Lug Nuts
- Use the wrench to fully tighten all lug nuts in a star pattern
- Apply significant pressure to ensure they're secure

### 10. Store Equipment
- Return the flat tire, jack, and tools to your vehicle

⚠️ **FINAL WARNING**: Have your regular tire repaired or replaced as soon as possible. Most spare tires are not designed for extended use or high speeds."""
    ),
    ResponseType(
        id="creative",
        name="Creative Content",
        description="Original, imaginative content with narrative elements.",
        config_options={
            "style": {
                "type": "select",
                "options": ["narrative", "poetic", "descriptive", "dialogue"],
                "default": "narrative",
                "description": "Style of creative content"
            },
            "tone": {
                "type": "select",
                "options": ["formal", "casual", "humorous", "dramatic", "inspirational"],
                "default": "casual",
                "description": "Tone of the content"
            }
        },
        example_output="""# The Silent Observer

The café hummed with the gentle percussion of morning routines. Steam hissed from the espresso machine, spoons clinked against ceramic, and keyboards clicked under hurried fingers. By the window, bathed in the honeyed light of early autumn, sat a woman who wasn't typing or talking or even drinking. She was watching.

Her eyes—sharp, deep, the color of over-steeped tea—moved methodically across the room, lingering on details others might miss: the barista's collection of mismatched rings, the architectural student erasing the same corner of his drawing for the third time, the elderly man whose trembling hands struggled with his teabag but whose face remained a portrait of dignity.

She had been coming here every Tuesday for six months. The staff had stopped asking for her order; a plain black coffee would appear at her table without request. No one knew her name or her story. Some regulars had constructed theories: she was a writer gathering character inspiration, a detective on an impossibly long stakeout, or perhaps someone navigating tremendous grief who found comfort in the proximity of strangers carrying on with their ordinary lives.

What none of them suspected was the truth: that in her pocket was a small notebook filled with observations of human kindness—the young professional who always held the door, the student who organized scattered sugar packets, the businessman who discreetly paid for a struggling mother's breakfast. Every Tuesday, she documented evidence that despite headlines and heartbreaks, compassion remained in the world's small corners.

And on particularly dark days, she would read her notes and remember why she had chosen to stay in it."""
    )
]

# Group response types by category
RESPONSE_TYPE_CATEGORIES = {
    "Basic": ["default", "json_mode"],
    "Reasoning": ["reasoning", "analytical"],
    "Specialized": ["instructional", "creative"]
}

# Map of response types by ID for easy lookup
RESPONSE_TYPES_BY_ID = {rt.id: rt for rt in RESPONSE_TYPES}


def response_type_selector() -> Dict:
    """
    UI for selecting and configuring response types.
    
    Returns:
        Configuration dictionary for the selected response type
    """
    st.subheader("Response Type")
    
    # Create tabs for response type categories
    category_tabs = st.tabs(list(RESPONSE_TYPE_CATEGORIES.keys()))
    
    selected_type_id = st.session_state.get("selected_response_type_id", "default")
    config = st.session_state.get("response_type_config", {})
    
    # Display response types by category
    for i, (category, type_ids) in enumerate(RESPONSE_TYPE_CATEGORIES.items()):
        with category_tabs[i]:
            # Get response types in this category
            category_types = [rt for rt in RESPONSE_TYPES if rt.id in type_ids]
            
            for response_type in category_types:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**{response_type.name}**")
                    st.markdown(response_type.description)
                
                with col2:
                    if st.button("Select", key=f"select_{response_type.id}"):
                        selected_type_id = response_type.id
                        # Initialize config with defaults if first time selecting
                        if selected_type_id not in config:
                            config[selected_type_id] = {}
                            for option_id, option in response_type.config_options.items():
                                config[selected_type_id][option_id] = option.get("default", None)
                        
                        st.session_state.selected_response_type_id = selected_type_id
                        st.session_state.response_type_config = config
                        st.rerun()
    
    # Display configuration for selected type
    st.markdown("---")
    selected_type = RESPONSE_TYPES_BY_ID.get(selected_type_id)
    
    if selected_type:
        st.markdown(f"### Selected: {selected_type.name}")
        st.markdown(selected_type.description)
        
        # Configuration options
        if selected_type.config_options:
            st.subheader("Configuration")
            
            for option_id, option in selected_type.config_options.items():
                option_type = option.get("type", "text")
                description = option.get("description", "")
                
                # Ensure the option exists in config
                if selected_type_id not in config:
                    config[selected_type_id] = {}
                if option_id not in config[selected_type_id]:
                    config[selected_type_id][option_id] = option.get("default", None)
                
                # Create the appropriate input widget
                if option_type == "text":
                    config[selected_type_id][option_id] = st.text_input(
                        f"{option_id}: {description}",
                        value=config[selected_type_id][option_id] or "",
                        key=f"config_{selected_type_id}_{option_id}"
                    )
                elif option_type == "select":
                    options = option.get("options", [])
                    config[selected_type_id][option_id] = st.selectbox(
                        f"{option_id}: {description}",
                        options=options,
                        index=options.index(config[selected_type_id][option_id]) if config[selected_type_id][option_id] in options else 0,
                        key=f"config_{selected_type_id}_{option_id}"
                    )
                elif option_type == "checkbox":
                    config[selected_type_id][option_id] = st.checkbox(
                        f"{option_id}: {description}",
                        value=bool(config[selected_type_id][option_id]),
                        key=f"config_{selected_type_id}_{option_id}"
                    )
                elif option_type == "number":
                    config[selected_type_id][option_id] = st.number_input(
                        f"{option_id}: {description}",
                        value=float(config[selected_type_id][option_id] or 0),
                        key=f"config_{selected_type_id}_{option_id}"
                    )
                elif option_type == "multiselect":
                    options = option.get("options", [])
                    default = config[selected_type_id][option_id] or option.get("default", [])
                    config[selected_type_id][option_id] = st.multiselect(
                        f"{option_id}: {description}",
                        options=options,
                        default=default,
                        key=f"config_{selected_type_id}_{option_id}"
                    )
        
        # Example output
        if selected_type.example_output:
            with st.expander("Show Example Output"):
                st.markdown(selected_type.example_output)
        
        # Update session state
        st.session_state.response_type_config = config
        
        # Return the selected type and config
        return {
            "type": selected_type_id,
            "config": config.get(selected_type_id, {})
        }
    
    return {"type": "default", "config": {}}


def reasoning_model_settings() -> Dict:
    """
    UI for configuring reasoning model settings.
    
    Returns:
        Dictionary of reasoning settings
    """
    st.subheader("Reasoning Settings")
    
    settings = st.session_state.get("reasoning_settings", {})
    
    # Initialize default settings if not present
    if not settings:
        settings = {
            "reasoning_depth": "standard",
            "think_step_by_step": True,
            "show_work": True,
            "use_numbered_steps": True,
            "max_reasoning_steps": 5,
            "evidence_threshold": "moderate"
        }
        st.session_state.reasoning_settings = settings
    
    # Create settings UI
    col1, col2 = st.columns(2)
    
    with col1:
        settings["reasoning_depth"] = st.select_slider(
            "Reasoning Depth",
            options=["minimal", "standard", "comprehensive", "exhaustive"],
            value=settings["reasoning_depth"]
        )
        
        settings["think_step_by_step"] = st.checkbox(
            "Think Step by Step",
            value=settings["think_step_by_step"],
            help="Instruct the model to break down reasoning into explicit steps"
        )
        
        settings["show_work"] = st.checkbox(
            "Show Work",
            value=settings["show_work"],
            help="Include intermediate calculations and reasoning in the response"
        )
    
    with col2:
        settings["use_numbered_steps"] = st.checkbox(
            "Use Numbered Steps",
            value=settings["use_numbered_steps"],
            help="Present reasoning as an ordered list of steps"
        )
        
        settings["max_reasoning_steps"] = st.slider(
            "Maximum Reasoning Steps",
            min_value=1,
            max_value=10,
            value=settings["max_reasoning_steps"],
            help="Maximum number of reasoning steps to include"
        )
        
        settings["evidence_threshold"] = st.select_slider(
            "Evidence Threshold",
            options=["minimal", "moderate", "substantial", "rigorous"],
            value=settings["evidence_threshold"],
            help="Level of evidence required for conclusions"
        )
    
    # Update session state
    st.session_state.reasoning_settings = settings
    
    return settings


def generate_reasoning_prompt(settings: Dict, user_prompt: str) -> str:
    """
    Generate a system instruction that encourages structured reasoning.
    
    Args:
        settings: Reasoning settings
        user_prompt: Original user prompt
        
    Returns:
        System instruction for reasoning
    """
    # Base instruction
    instruction = "You are an assistant that provides thorough, well-reasoned responses. "
    
    # Add step-by-step thinking instruction
    if settings.get("think_step_by_step", True):
        instruction += "Think step-by-step to solve problems systematically. "
    
    # Add depth instruction
    depth = settings.get("reasoning_depth", "standard")
    if depth == "minimal":
        instruction += "Provide concise reasoning with only essential steps. "
    elif depth == "standard":
        instruction += "Provide clear reasoning with key steps and considerations. "
    elif depth == "comprehensive":
        instruction += "Provide detailed reasoning exploring multiple facets of the problem. "
    elif depth == "exhaustive":
        instruction += "Provide exhaustive reasoning covering all aspects and edge cases. "
    
    # Add show work instruction
    if settings.get("show_work", True):
        instruction += "Show your work and intermediate steps. "
    
    # Add numbered steps instruction
    if settings.get("use_numbered_steps", True):
        instruction += "Present your reasoning as a numbered list of steps. "
    
    # Add evidence threshold instruction
    evidence = settings.get("evidence_threshold", "moderate")
    if evidence == "minimal":
        instruction += "Focus on reaching a conclusion efficiently. "
    elif evidence == "moderate":
        instruction += "Support key points with relevant evidence. "
    elif evidence == "substantial":
        instruction += "Provide substantial evidence for all significant claims. "
    elif evidence == "rigorous":
        instruction += "Apply rigorous standards of evidence and address potential counterarguments. "
    
    # Add max steps instruction
    max_steps = settings.get("max_reasoning_steps", 5)
    instruction += f"Limit your reasoning to approximately {max_steps} key steps. "
    
    return instruction


def response_types_ui():
    """Main UI component for response types and reasoning models."""
    st.header("Response Types & Reasoning Models")
    
    tab1, tab2 = st.tabs(["Response Type Selector", "Reasoning Settings"])
    
    with tab1:
        response_config = response_type_selector()
        
        # Save to session state for use in playground
        if "response_config" not in st.session_state:
            st.session_state.response_config = response_config
        else:
            st.session_state.response_config = response_config
    
    with tab2:
        reasoning_settings = reasoning_model_settings()
        
        # Example user prompt
        user_prompt = st.text_area(
            "Test Prompt",
            value="Explain the significance of the Drake equation in estimating the number of active, communicative extraterrestrial civilizations in our galaxy."
        )
        
        # Generate and show reasoning instruction
        if st.button("Generate Reasoning Instruction"):
            instruction = generate_reasoning_prompt(reasoning_settings, user_prompt)
            
            st.subheader("System Instruction for Reasoning")
            st.code(instruction)
            
            # Save to session state
            st.session_state.reasoning_instruction = instruction