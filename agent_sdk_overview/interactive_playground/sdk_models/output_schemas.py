"""
Output schema support for the SDK Playground.
This module provides components for defining, visualizing and testing output schemas.
"""
import streamlit as st
import json
from typing import Dict, Any, Optional, List, Union
import jsonschema
from uuid import uuid4


class SchemaDefinition:
    """Schema definition with validation and example generation capabilities."""
    
    SCHEMA_TEMPLATES = {
        "simple_message": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "A simple text message"
                }
            },
            "required": ["message"]
        },
        "structured_analysis": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A brief summary of the analysis"
                },
                "points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "The title of the point"
                            },
                            "description": {
                                "type": "string",
                                "description": "A detailed description of the point"
                            },
                            "importance": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5,
                                "description": "The importance of the point on a scale of 1-5"
                            }
                        },
                        "required": ["title", "description"]
                    },
                    "description": "A list of key points in the analysis"
                },
                "conclusion": {
                    "type": "string",
                    "description": "A conclusion based on the analysis"
                }
            },
            "required": ["summary", "points", "conclusion"]
        },
        "sentiment_analysis": {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "neutral", "negative"],
                    "description": "The overall sentiment of the text"
                },
                "score": {
                    "type": "number",
                    "minimum": -1,
                    "maximum": 1,
                    "description": "The sentiment score from -1 (negative) to 1 (positive)"
                },
                "aspects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "aspect": {
                                "type": "string",
                                "description": "The aspect being analyzed"
                            },
                            "sentiment": {
                                "type": "string",
                                "enum": ["positive", "neutral", "negative"],
                                "description": "The sentiment for this aspect"
                            }
                        },
                        "required": ["aspect", "sentiment"]
                    },
                    "description": "Sentiment analysis for specific aspects of the text"
                }
            },
            "required": ["sentiment", "score"]
        },
        "categorical_classification": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "The primary category for the content"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Confidence score for the classification"
                },
                "alternative_categories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "Alternative category"
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "description": "Confidence score for this category"
                            }
                        },
                        "required": ["category", "confidence"]
                    },
                    "description": "List of alternative categories with confidence scores"
                }
            },
            "required": ["category", "confidence"]
        },
        "custom": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }

    def __init__(self, schema: Dict = None, template_name: str = None):
        """
        Initialize with either a schema dict or a template name.
        
        Args:
            schema: A JSON schema dict
            template_name: Name of a predefined template
        """
        if schema:
            self.schema = schema
        elif template_name and template_name in self.SCHEMA_TEMPLATES:
            self.schema = self.SCHEMA_TEMPLATES[template_name]
        else:
            self.schema = self.SCHEMA_TEMPLATES["custom"]
        
        self.id = str(uuid4())
    
    def validate(self, data: Dict) -> bool:
        """
        Validate data against the schema.
        
        Args:
            data: The data to validate
            
        Returns:
            True if valid, raises jsonschema.ValidationError if invalid
        """
        try:
            jsonschema.validate(instance=data, schema=self.schema)
            return True
        except jsonschema.ValidationError:
            raise
    
    def generate_example(self) -> Dict:
        """
        Generate an example object that conforms to the schema.
        
        Returns:
            A dict with example values
        """
        return self._generate_for_schema(self.schema)
    
    def _generate_for_schema(self, schema: Dict) -> Any:
        """
        Recursively generate example values for a schema.
        
        Args:
            schema: A JSON schema or subschema
            
        Returns:
            An example value conforming to the schema
        """
        if "enum" in schema:
            # For enum types, return the first value
            return schema["enum"][0]
        
        schema_type = schema.get("type")
        
        if schema_type == "object":
            result = {}
            properties = schema.get("properties", {})
            
            for prop_name, prop_schema in properties.items():
                result[prop_name] = self._generate_for_schema(prop_schema)
            
            return result
        
        elif schema_type == "array":
            # Generate an array with one example item
            items_schema = schema.get("items", {})
            return [self._generate_for_schema(items_schema)]
        
        elif schema_type == "string":
            # Use description as example if available
            return schema.get("description", "Example string")
        
        elif schema_type == "integer" or schema_type == "number":
            minimum = schema.get("minimum", 0)
            maximum = schema.get("maximum", 10)
            # Use the minimum or a default
            return minimum
        
        elif schema_type == "boolean":
            return True
        
        # Default for unrecognized types
        return None


def schema_builder_ui() -> Optional[Dict]:
    """
    UI for building and editing JSON schemas.
    
    Returns:
        The schema dictionary if finalized, or None
    """
    st.subheader("Output Schema Builder")
    
    # Template selection
    template_options = list(SchemaDefinition.SCHEMA_TEMPLATES.keys())
    template_names = [
        "Simple Message",
        "Structured Analysis",
        "Sentiment Analysis",
        "Categorical Classification",
        "Custom Schema"
    ]
    template_display = {k: v for k, v in zip(template_options, template_names)}
    
    selected_template = st.selectbox(
        "Select a schema template:",
        options=template_options,
        format_func=lambda x: template_display[x]
    )
    
    # Initialize schema from template
    if "current_schema" not in st.session_state or st.session_state.get("schema_template") != selected_template:
        schema_def = SchemaDefinition(template_name=selected_template)
        st.session_state.current_schema = schema_def.schema
        st.session_state.schema_template = selected_template
    
    # Schema editor
    schema_str = st.text_area(
        "Edit schema (JSON):",
        value=json.dumps(st.session_state.current_schema, indent=2),
        height=300
    )
    
    # Try to parse the schema
    try:
        edited_schema = json.loads(schema_str)
        st.session_state.current_schema = edited_schema
        schema_valid = True
    except json.JSONDecodeError:
        st.error("Invalid JSON. Please check your syntax.")
        schema_valid = False
    
    # If schema is valid, show example
    if schema_valid:
        try:
            schema_def = SchemaDefinition(schema=edited_schema)
            example = schema_def.generate_example()
            
            with st.expander("Example Output"):
                st.code(json.dumps(example, indent=2), language="json")
            
            # Option to use this schema
            if st.button("Use This Schema"):
                return edited_schema
        except Exception as e:
            st.error(f"Error generating example: {str(e)}")
    
    return None


def schema_visualizer(schema: Dict):
    """
    Visualize a JSON schema structure.
    
    Args:
        schema: The JSON schema to visualize
    """
    def render_schema_node(schema_node, path="root"):
        """Recursively render a schema node."""
        if "type" not in schema_node:
            st.markdown(f"**{path}**: _(no type specified)_")
            return
        
        node_type = schema_node["type"]
        description = schema_node.get("description", "")
        
        if node_type == "object":
            st.markdown(f"**{path}** (object): {description}")
            properties = schema_node.get("properties", {})
            required = schema_node.get("required", [])
            
            for prop_name, prop_schema in properties.items():
                # Indicate if property is required
                req_mark = "*" if prop_name in required else ""
                render_schema_node(prop_schema, f"{path}.{prop_name}{req_mark}")
        
        elif node_type == "array":
            st.markdown(f"**{path}** (array): {description}")
            items_schema = schema_node.get("items", {})
            render_schema_node(items_schema, f"{path}[]")
        
        else:
            # For primitive types
            enum_values = schema_node.get("enum", [])
            enum_str = f", options: {', '.join(map(str, enum_values))}" if enum_values else ""
            st.markdown(f"**{path}** ({node_type}): {description}{enum_str}")
    
    st.subheader("Schema Structure")
    render_schema_node(schema)


def schema_support_ui():
    """Main UI component for schema support."""
    st.header("Output Schema Support")
    
    tab1, tab2 = st.tabs(["Schema Builder", "Schema Testing"])
    
    with tab1:
        schema = schema_builder_ui()
        if schema:
            schema_visualizer(schema)
            
            # Save schema to session state for use in the playground
            st.session_state.output_schema = schema
            st.success("Schema saved and ready to use in the playground.")
    
    with tab2:
        # Schema validation testing
        if "output_schema" in st.session_state:
            st.subheader("Schema Validation Testing")
            
            # Show current schema
            st.code(json.dumps(st.session_state.output_schema, indent=2), language="json")
            
            # Text area for input to validate
            test_input = st.text_area(
                "Enter JSON to validate against the schema:",
                value=json.dumps(SchemaDefinition(schema=st.session_state.output_schema).generate_example(), indent=2),
                height=300
            )
            
            if st.button("Validate"):
                try:
                    test_data = json.loads(test_input)
                    schema_def = SchemaDefinition(schema=st.session_state.output_schema)
                    
                    try:
                        schema_def.validate(test_data)
                        st.success("✅ Valid! The input conforms to the schema.")
                    except jsonschema.ValidationError as e:
                        st.error(f"❌ Validation failed: {e.message}")
                        st.markdown(f"Path: `{'.'.join(map(str, e.path))}`")
                except json.JSONDecodeError:
                    st.error("Invalid JSON. Please check your syntax.")
        else:
            st.info("Please build and save a schema in the Schema Builder tab first.")