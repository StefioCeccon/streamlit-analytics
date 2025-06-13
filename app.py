import streamlit as st
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Create a logger
logger = logging.getLogger(__name__)

logger.info("App is running")

# Main page content
st.title("Welcome to Xpertus")
st.write("Navigate to different sections using the sidebar menu.")