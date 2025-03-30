"""
Wrapper for showcase UI to handle potential import issues.
This provides a clean integration point for the main app.
"""
import streamlit as st

# Import the showcase UI with proper error handling
def run_showcase():
    """Run the showcase UI with proper error handling."""
    try:
        from .showcase import showcase_ui
        showcase_ui()
    except ImportError as e:
        st.error(f"Error loading Advanced Playground: {e}")
        st.markdown("""
        ### Troubleshooting
        
        The Advanced Playground features could not be loaded. This might be due to:
        
        1. Missing dependencies - Try running `pip install jsonschema`
        2. Import path issues - Make sure the directory structure is correct
        3. Missing files - Check that all required files are present in the sdk_models directory
        
        Please see the error message above for more details.
        """)
    except Exception as e:
        st.error(f"Error running Advanced Playground: {e}")
        import traceback
        st.code(traceback.format_exc())