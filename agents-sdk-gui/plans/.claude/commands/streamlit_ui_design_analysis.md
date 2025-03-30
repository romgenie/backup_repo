/streamlit_ui_design_analysis [SOURCE_DOCUMENTATION] [STREAMLIT_VERSION] [TARGET_APPLICATION]
# Streamlit UI Design Analysis

## Task Description
Review the Streamlit documentation to identify essential components for designing a user interface using Streamlit within [TARGET_APPLICATION]. Extract relevant sections from the documentation and create a comprehensive UI design guide tailored to Streamlit’s features.

## Input Parameters
- SOURCE_DOCUMENTATION: URL or local path to the Streamlit documentation
- STREAMLIT_VERSION: Version of Streamlit to be used (e.g., 1.10.0)
- TARGET_APPLICATION: Description or path to the application where the Streamlit UI will be implemented

## Required Outputs

1. Analysis Summary:
   - Key Streamlit components (interactive widgets, layout management, state handling, theming) relevant to [TARGET_APPLICATION]
   - UI design integration points leveraging Streamlit’s capabilities and customization options
   - Potential challenges (e.g., performance optimization, responsive design) and recommendations for implementing a Streamlit-based UI

2. Documentation Extraction Plan:
   - List of essential sections from the Streamlit documentation to extract (e.g., widget catalog, layout configuration, component customization, deployment considerations)
   - Justification for each selection based on the UI design needs and integration with [TARGET_APPLICATION]

3. Extraction Script:
   - A Python script that extracts UI-related documentation content from SOURCE_DOCUMENTATION
   - Organizes the extracted content into a logical structure (e.g., sections for widgets, layout, theming)
   - Accepts command-line parameters for source location and customization options
   - Handles errors gracefully and outputs clear status messages

4. Streamlit UI Design Guide Generation Script:
   - A Python script that generates a custom UI design guide for integrating Streamlit into [TARGET_APPLICATION]
   - Includes practical code examples demonstrating the use of Streamlit APIs (e.g., st.button, st.sidebar, st.write, st.columns)
   - Tailors the guide to the architectural patterns and requirements of [TARGET_APPLICATION]
   - Provides recommendations and best practices for building a responsive and maintainable UI with Streamlit

## Script Requirements
- Both scripts should be modular, well-documented, and reusable
- Include command-line arguments for user customization
- Gracefully handle error conditions and provide clear, informative output on actions taken
- Incorporate inline comments and documentation for ease of maintenance

## Example Usage
/streamlit_ui_design_analysis /path/to/streamlit/documentation 1.10.0 /path/to/target/application

## Alternate Usage
/streamlit_ui_design_analysis [SOURCE_DOCUMENTATION] [STREAMLIT_VERSION] [TARGET_APPLICATION]

