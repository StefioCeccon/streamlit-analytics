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

# Define the pages

logger.info("App is running")

analytics_page = st.Page("analytics.py", title="Analytics", icon="📱")
# page_2 = st.Page("page2.py", title="Page 2", icon="🎉")

# main_page = st.Page("home.py", title="Main Page", icon="🎈")
# page_2 = st.Page("page2.py", title="Page 2", icon="🎉")

# Set up navigation
pg = st.navigation([analytics_page])

# Run the selected page
pg.run()