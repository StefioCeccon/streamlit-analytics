import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import json
import os
from dotenv import load_dotenv
import html

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

db_user_users = os.getenv('APP_POSTGRES_USER')
db_password_users = os.getenv('APP_POSTGRES_PASSWORD')
db_host_users = os.getenv('APP_POSTGRES_HOST')
db_port_users = os.getenv('APP_POSTGRES_PORT')
db_name_users = os.getenv('APP_POSTGRES_DATABASE')

# Construct connection string from environment variables
connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode={db_ssl_mode}"

connection_string_users_db = f"postgresql://{db_user_users}:{db_password_users}@{db_host_users}:{db_port_users}/{db_name_users}?sslmode={db_ssl_mode}"

@st.cache_resource
def get_connection():
    return create_engine(connection_string)

@st.cache_resource
def get_users_connection():
    return create_engine(connection_string_users_db)

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

# Query to get agency and user information
def get_agency_user_info():
    engine = get_users_connection()
    
    # Get agency information
    agency_query = text("""
        SELECT id, name as agency_name
        FROM agencies
    """)
    agencies_df = pd.read_sql(agency_query, engine)
    
    # Get user information
    users_query = text("""
        SELECT aau.provider_user_id, aau.username, aau.name
        FROM agency_account_users aau
    """)
    users_df = pd.read_sql(users_query, engine)
    
    return agencies_df, users_df

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
agencies_df, users_df = get_agency_user_info()

# Create a mapping for agency names
agency_mapping = dict(zip(agencies_df['id'], agencies_df['agency_name']))
# Create a mapping for user information
user_mapping = dict(zip(users_df['provider_user_id'], 
                       users_df.apply(lambda x: f"{x['name']} ({x['username']})", axis=1)))

# Initialize session state if not exists
if 'show_all_messages' not in st.session_state:
    st.session_state.show_all_messages = False

# Callback function to update session state
def toggle_callback():
    st.session_state.show_all_messages = not st.session_state.show_all_messages

# Create a select box for agencies with friendly names
unique_agencies = df['agency_id'].unique()
agency_options = {
    f"{agency_mapping.get(agency_id, f'Agency {agency_id}')} {'(ID: ' + str(agency_id) + ')' if st.session_state.show_all_messages else ''}": agency_id 
    for agency_id in unique_agencies
}
selected_agency_name = st.selectbox('Select Agency', list(agency_options.keys()))
selected_agency = agency_options[selected_agency_name]

# Filter data for selected agency
agency_data = df[df['agency_id'] == selected_agency]

# Create a select box for senders with friendly names
unique_senders = agency_data['provider_sender_id'].unique()
sender_options = {
    f"{user_mapping.get(sender_id, f'User {sender_id}')} {'(ID: ' + str(sender_id) + ')' if st.session_state.show_all_messages else ''}": sender_id 
    for sender_id in unique_senders
}
selected_sender_name = st.selectbox('Select Sender', list(sender_options.keys()))
selected_sender = sender_options[selected_sender_name]

# Filter data for selected sender
chat_data = agency_data[agency_data['provider_sender_id'] == selected_sender]

# Add toggle for showing all messages
st.toggle('Show all messages (including internal AI messages and IDs)', 
          value=st.session_state.show_all_messages,
          on_change=toggle_callback)

# Display chat messages
st.markdown("### Chat History")
for _, row in chat_data.iterrows():
    messages = parse_chat_messages(row['state_data'])
    
    # Filter messages if not showing all
    if not st.session_state.get('show_all_messages', False):
        messages = [msg for msg in messages if msg['name'] in ['user_to_main_router', 'formatter_to_user']]
    
    if messages:  # Only show container if there are messages to display
        # Create a container for each chat session
        with st.container():
            st.markdown(f"**Chat from {row['provider']} - {row['updated_at']}**")
            
            # Display messages in a chat-like format
            for msg in messages:
                name_html = (
                    f"<div style='font-size: 0.85em; color: #888; margin-bottom: 3px;'><strong>Name:</strong> {msg['name']}</div>"
                    if st.session_state.get('show_all_messages', False) else ""
                )
                if msg['type'] == 'human':
                    st.markdown(
                        f"<div style='background-color: #e6f3ff; padding: 10px; border-radius: 10px; margin: 5px;'>"
                        f"{name_html}"
                        f"{msg['content']}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px; margin: 5px;'>"
                        f"{name_html}"
                        f"{msg['content']}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            
            st.divider()