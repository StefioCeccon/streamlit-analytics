import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# st.markdown("# Chats 💬")
# st.sidebar.markdown("# 💬 Chat History")
# st.sidebar.markdown("---")

# Get database credentials from environment variables
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')
db_ssl_mode = os.getenv('DB_SSL_MODE')

# Construct connection string from environment variables
connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode={db_ssl_mode}"

@st.cache_resource
def get_connection():
    return create_engine(connection_string)

# Query to get chat data
def get_chat_data():
    engine = get_connection()
    query = text("""
        SELECT agency_id, provider_sender_id, provider, state_data, updated_at
        FROM travelbot_states
        UNION ALL
        SELECT agency_id, provider_sender_id, provider, state_data, updated_at
        FROM travelbot_states_history
        ORDER BY updated_at DESC
    """)
    return pd.read_sql(query, engine)

# Function to parse chat messages
def parse_chat_messages(state_data):
    try:
        messages = state_data['messages']
        print(messages)
        parsed_messages = []
        for msg in messages:
            if msg.get('type') in ['human', 'ai']:
                parsed_messages.append({
                    'content': msg.get('content', ''),
                    'type': msg.get('type', ''),
                    'name': msg.get('name', 'Unknown')
                })
        return parsed_messages
    except:
        return []

# Get the data
df = get_chat_data()

# Create a select box for agencies
unique_agencies = df['agency_id'].unique()
selected_agency = st.selectbox('Select Agency', unique_agencies)

# Filter data for selected agency
agency_data = df[df['agency_id'] == selected_agency]

# Create a select box for senders
unique_senders = agency_data['provider_sender_id'].unique()
selected_sender = st.selectbox('Select Sender', unique_senders)

# Filter data for selected sender
chat_data = agency_data[agency_data['provider_sender_id'] == selected_sender]

# Display chat messages
st.markdown("### Chat History")
for _, row in chat_data.iterrows():
    messages = parse_chat_messages(row['state_data'])
    # Create a container for each chat session
    with st.container():
        st.markdown(f"**Chat from {row['provider']} - {row['updated_at']}**")
        
        # Display messages in a chat-like format
        for msg in messages:
            if msg['type'] == 'human':
                st.markdown(f"""
                    <div style='background-color: #e6f3ff; padding: 10px; border-radius: 10px; margin: 5px;'>
                        <strong>User:</strong> {msg['content']}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px; margin: 5px;'>
                        <strong>AI:</strong> {msg['content']}
                    </div>
                """, unsafe_allow_html=True)
        
        st.divider()