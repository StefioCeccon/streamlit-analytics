import streamlit as st
import numpy as np
import pandas as pd
import time
import os
from sqlalchemy import create_engine
import plotly.express as px
import altair as alt
import datetime
import dotenv
import logging
import openai
from openai import OpenAI

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Xpertus Analytics",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
<style>
[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    text-align: center;
    padding: 15px 0;
    border-radius: 5px;
    color: #333333;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
  color: #333333;
  font-weight: 600;
}

[data-testid="stMetricValue"] {
  color: #333333;
}

[data-testid="stMetricDelta"] {
  color: #333333;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}
</style>
""", unsafe_allow_html=True)

st.markdown("# Xpertus Usage Analytics 📱")
st.sidebar.markdown("# Analytics 📱")

## set up db

# Create connection string
connection_string = os.environ.get("LOGGER_URL_STRING")

# Comment out database connection
# @st.cache_resource
# def get_connection():
#     logger.info("Creating connection")
#     return create_engine(connection_string)

# # Connect to database
# conn = get_connection()
# logger.info(conn)

# Generate mock data instead of fetching from database
def generate_mock_data():
    logger.info("Generating mock data")
    
    # Create date range for the last 90 days
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=90)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create user IDs
    user_ids = [f"user_{i}" for i in range(1, 101)]
    
    # Create events for travel agency Instagram bot
    events = ["Conversation_Start", "Destination_Query", "Price_Query", "Booking_Request", 
              "Booking_Confirmed", "Itinerary_Request", "Support_Query", "Feedback"]
    
    # Create flavors (keep original column name for compatibility)
    flavors = ["Instagram", "Website", "Mobile_App"]
    
    # Create OS versions (repurposed for user demographics)
    age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
    
    # Create app versions (repurposed for trip types)
    trip_types = ["Beach", "City", "Adventure", "Cultural", "Luxury", "Budget", "Family"]
    
    # Create device models (repurposed for destinations)
    destinations = ["Paris", "Bali", "New York", "Tokyo", "Cancun", "London", 
                   "Barcelona", "Dubai", "Rome", "Maldives", "Sydney"]
    
    # Generate random data
    rows = []
    for _ in range(10000):  # Generate 10,000 rows of mock data
        date = np.random.choice(dates)
        user_id = np.random.choice(user_ids)
        event = np.random.choice(events)
        flavor = np.random.choice(flavors, p=[0.7, 0.2, 0.1])  # Instagram is more common
        
        # Repurpose OS and OSVersion for user demographics
        os = "Instagram Bot"
        os_version = np.random.choice(age_groups)
        
        # Repurpose clientVersion for trip types
        client_version = np.random.choice(trip_types)
        
        # Repurpose model for destinations
        if event in ["Destination_Query", "Price_Query", "Booking_Request", "Booking_Confirmed"]:
            model = np.random.choice(destinations)
        else:
            model = "Unspecified"
        
        # Add booking value as a custom field
        if event == "Booking_Confirmed":
            booking_value = int(np.random.normal(1500, 500))
        else:
            booking_value = 0
        
        rows.append({
            "clientDate": date,
            "userClientId": user_id,
            "event": event,
            "flavor": flavor,
            "OS": os,
            "OSVersion": os_version,
            "clientVersion": client_version,
            "model": model,
            "booking_value": booking_value
        })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Add some trends to make the data more realistic
    # More bookings on weekends
    weekend_mask = df["clientDate"].dt.dayofweek >= 5  # Saturday and Sunday
    df.loc[weekend_mask, "event"] = np.random.choice(
        events, 
        size=sum(weekend_mask), 
        p=[0.15, 0.15, 0.15, 0.2, 0.15, 0.1, 0.05, 0.05]
    )
    
    # More beach destinations in summer months (assuming Northern Hemisphere)
    summer_mask = (df["clientDate"].dt.month >= 6) & (df["clientDate"].dt.month <= 8)
    beach_destinations = ["Bali", "Cancun", "Maldives"]
    summer_dest_mask = summer_mask & df["model"].isin(beach_destinations)
    if sum(summer_dest_mask) > 0:
        df.loc[summer_dest_mask, "booking_value"] = df.loc[summer_dest_mask, "booking_value"] * 1.2
    
    logger.info(f"Generated mock data with shape: {df.shape}")
    return df

# Use mock data instead of database query
def fetch_all_data():
    try:
        return generate_mock_data()
    except Exception as e:
        logger.error(f"Error generating mock data: {e}")
        st.error(f"Error generating mock data: {e}")
        return pd.DataFrame()

# Load all data once
try:
    logger.info("Attempting to fetch mock data")
    all_data = fetch_all_data()
    logger.info(f"Loaded data shape: {all_data.shape}")
    
    if all_data.empty:
        logger.warning("No mock data available")
        st.warning("No data available for testing.")
except Exception as e:
    logger.error(f"Error loading mock data: {e}")
    st.error(f"Error loading mock data: {e}")
    all_data = pd.DataFrame()  # Create empty DataFrame to prevent further errors

# Get min and max dates from the data
if not all_data.empty:
    min_date = all_data["clientDate"].min().date()
    max_date = all_data["clientDate"].max().date()
else:
    min_date = datetime.date.today() - datetime.timedelta(days=30)
    max_date = datetime.date.today()

# Sidebar filters
with st.sidebar:
    # Date range filter
    st.subheader("Date Range")
    
    start_date = st.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("End date", max_date, min_value=min_date, max_value=max_date)
    
    # Flavor filter
    all_flavors = sorted(all_data["flavor"].unique().tolist())
    selected_flavors = st.multiselect("Select flavors", all_flavors, default=all_flavors)
    
    # Color theme - REMOVED
    # color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    # selected_color_theme = st.selectbox('Select a color theme', color_theme_list)

# Filter data based on selections
@st.cache_data(ttl=3600, max_entries=10)
def filter_data(data, start_date, end_date, flavors):
    if data.empty:
        return pd.DataFrame()
    
    try:
        filtered = data.copy()
        filtered = filtered[(filtered["clientDate"].dt.date >= start_date) & 
                            (filtered["clientDate"].dt.date <= end_date)]
        
        if flavors:
            filtered = filtered[filtered["flavor"].isin(flavors)]
        
        # Log filtered data size
        logger.info(f"Filtered data shape: {filtered.shape}")
        return filtered
    except Exception as e:
        logger.error(f"Error filtering data: {e}")
        st.error(f"Error filtering data: {e}")
        return pd.DataFrame()

# Apply filters
filtered_data = filter_data(all_data, start_date, end_date, selected_flavors)

# Main dashboard layout
col = st.columns((1.5, 4.5, 2), gap='medium')

# Left column - Key metrics
with col[0]:
    st.markdown('#### Key Metrics')
    
    # Total users
    total_users = filtered_data["userClientId"].nunique()
    
    # Total conversations
    total_conversations = filtered_data[filtered_data["event"] == "Conversation_Start"].shape[0]
    
    # Total bookings
    total_bookings = filtered_data[filtered_data["event"] == "Booking_Confirmed"].shape[0]
    
    # Display metrics
    st.metric("Total Users", total_users)
    st.metric("Total Conversations", total_conversations)
    st.metric("Total Bookings", total_bookings)

# Middle column - Destination Trends chart (replacing iOS Version chart)
with col[1]:
    st.markdown('#### Destination Trends Over Time')
    
    # Process destination data (using model field)
    if not filtered_data.empty and "model" in filtered_data.columns:
        # Filter for rows with actual destinations (not "Unspecified")
        dest_data = filtered_data[filtered_data["model"] != "Unspecified"].copy()
        dest_data["date"] = dest_data["clientDate"].dt.strftime('%Y-%m-%d')
        
        # Get top 5 destinations
        top_destinations = dest_data["model"].value_counts().nlargest(5).index.tolist()
        dest_data = dest_data[dest_data["model"].isin(top_destinations)]
        
        # Group by date and destination
        dest_trend_data = dest_data.groupby(["date", "model"]).size().reset_index(name="count")
        
        if not dest_trend_data.empty:
            # Convert date to datetime for proper sorting and rolling average
            dest_trend_data['date'] = pd.to_datetime(dest_trend_data['date'])
            
            # Create a pivot table for easier rolling average calculation
            pivot_data = dest_trend_data.pivot(index='date', columns='model', values='count').fillna(0)
            
            # Apply 5-day rolling average
            smoothed_data = pivot_data.rolling(window=5, min_periods=1).mean()
            
            # Convert back to long format for plotting
            smoothed_data = smoothed_data.reset_index().melt(id_vars='date', var_name='destination', value_name='count')
            
            # Create line chart with smoothing
            fig = px.line(
                smoothed_data, 
                x='date', 
                y='count', 
                color='destination',
                title='Destination Trends Over Time (5-Day Average)',
                labels={'count': 'Number of Queries/Bookings', 'date': 'Date', 'destination': 'Destination'}
            )
            fig.update_traces(line_shape='linear')
            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                legend_title_text='Destination'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No destination data available for the selected filters.")

# Right column - Top data
with col[2]:
    st.markdown('#### Top Destinations')
    
    # Get destinations data from filtered data (using model field)
    try:
        if 'model' in filtered_data.columns:
            # Filter out "Unspecified" destinations
            dest_data = filtered_data[filtered_data['model'] != "Unspecified"]
            destinations_data = dest_data.groupby('model').size().reset_index(name='count')
            destinations_data = destinations_data.sort_values('count', ascending=False).head(10)
            
            if not destinations_data.empty:
                st.dataframe(
                    destinations_data,
                    column_order=("model", "count"),
                    hide_index=True,
                    width=None,
                    column_config={
                        "model": st.column_config.TextColumn("Destination"),
                        "count": st.column_config.ProgressColumn(
                            "Count",
                            format="%d",
                            min_value=0,
                            max_value=int(destinations_data["count"].max()) if not destinations_data.empty else 0,
                        )
                    }
                )
            else:
                st.info("No destination data available for the selected filters.")
        else:
            st.info("Destination data not available in the dataset.")
    except Exception as e:
        logger.error(f"Error displaying destinations: {e}")
        st.error(f"Error displaying destinations: {e}")

# Create a new row for Daily Booking Revenue
daily_users_col = st.columns([6, 2])

with daily_users_col[0]:
    st.markdown('#### Daily Booking Revenue')
    
    # Process daily booking revenue data
    booking_data = filtered_data[filtered_data['event'] == 'Booking_Confirmed'].copy()
    if not booking_data.empty:
        daily_revenue = booking_data.groupby(booking_data['clientDate'].dt.strftime('%Y-%m-%d'))['booking_value'].sum().reset_index()
        daily_revenue.columns = ['date', 'revenue']
        daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])
        
        # Sort by date to ensure proper rolling average
        daily_revenue = daily_revenue.sort_values('date')
        
        # Apply 5-day rolling average
        daily_revenue['revenue_smooth'] = daily_revenue['revenue'].rolling(window=5, min_periods=1).mean()
        
        fig = px.line(
            daily_revenue, 
            x='date', 
            y='revenue_smooth',
            title='Daily Booking Revenue (5-Day Average)',
            labels={'revenue_smooth': 'Revenue ($)', 'date': 'Date'}
        )
        fig.update_traces(line_shape='linear')
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No booking data available for the selected filters.")

with daily_users_col[1]:
    st.markdown('#### Top Trip Types')
    
    # Process trip types data (using clientVersion field)
    trip_types_data = filtered_data.groupby('clientVersion').size().reset_index(name='count')
    trip_types_data = trip_types_data.sort_values('count', ascending=False).head(10)
    
    if not trip_types_data.empty:
        st.dataframe(
            trip_types_data,
            column_order=("clientVersion", "count"),
            hide_index=True,
            width=None,
            column_config={
                "clientVersion": st.column_config.TextColumn("Trip Type"),
                "count": st.column_config.ProgressColumn(
                    "Count",
                    format="%d",
                    min_value=0,
                    max_value=int(trip_types_data["count"].max()),
                )
            }
        )

# Create another row for Event Distribution that spans columns 1 and 2
event_dist_col = st.columns([6, 2])

with event_dist_col[0]:
    st.markdown('#### Event Distribution')
    
    # Process events data
    events_data = filtered_data.groupby('event').size().reset_index(name='count')
    events_data = events_data.sort_values('count', ascending=False)
    
    if not events_data.empty:
        # Calculate percentage for each event
        total_events = events_data['count'].sum()
        events_data['percentage'] = events_data['count'] / total_events * 100
        
        # Add a column for text display - empty for small slices
        events_data['text'] = events_data.apply(
            lambda row: f"{row['event']}<br>{row['percentage']:.1f}%" 
            if row['percentage'] >= 1.0 else "", axis=1
        )
        
        fig = px.pie(
            events_data, 
            values='count', 
            names='event',
            title='Event Distribution',
            color_discrete_sequence=px.colors.sequential.Plasma,
            custom_data=['text']  # Include our custom text
        )
        
        fig.update_traces(
            textposition='inside',
            texttemplate='%{customdata[0]}',  # Use our custom text
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent:.1f}%<extra></extra>'
        )
        
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)'
        )
        st.plotly_chart(fig, use_container_width=True)

# Create a new row for the chat interface
st.markdown('#### Chat with Your Data 💬')
st.markdown('Ask any questions about the app usage data shown in this dashboard.')

# Initialize chat history
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Display chat messages from history
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to generate responses using OpenAI's ChatGPT
def generate_openai_response(prompt, data):
    try:
        # Get API key from environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
        
        if not api_key:
            return "Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
        
        # Create a client
        client = OpenAI(api_key=api_key)
        
        # Prepare context about the data
        data_context = f"""
        Current analytics data context:
        - Date range: {start_date} to {end_date}
        - Total users: {total_users}
        - Total conversations: {total_conversations}
        - Total bookings: {total_bookings}
        
        Top destinations: {', '.join([f"{dest}: {count}" for dest, count in filtered_data[filtered_data['model'] != 'Unspecified']['model'].value_counts().head(5).items()])}
        
        Top trip types: {', '.join([f"{trip}: {count}" for trip, count in filtered_data['clientVersion'].value_counts().head(5).items()])}
        
        Event distribution: {', '.join([f"{event}: {count}" for event, count in filtered_data['event'].value_counts().items()])}
        
        User age groups: {', '.join([f"{age}: {count}" for age, count in filtered_data['OSVersion'].value_counts().items()])}
        
        Total booking revenue: ${filtered_data[filtered_data['event'] == 'Booking_Confirmed']['booking_value'].sum()}
        """
        
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # You can use "gpt-4" for better results if you have access
            messages=[
                {"role": "system", "content": f"You are a helpful travel analytics assistant that answers questions about a travel agency's Instagram bot data. Use only the following context to answer questions. If you can't answer based on this context, say so. {data_context}"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        # Extract and return the response text
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        return f"Sorry, I encountered an error: {str(e)}"

# Add OpenAI API key input in the sidebar
with st.sidebar:
    st.divider()
    st.subheader("OpenAI API Settings")
    
    # Get API key from environment or let user input it
    default_api_key = os.environ.get("OPENAI_API_KEY", "")
    api_key = st.text_input("OpenAI API Key", value=default_api_key, type="password")
    
    # Save API key to environment variable
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

# Chat input
if prompt := st.chat_input("Ask a question about the data..."):
    # Add user message to chat history
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if os.environ.get("OPENAI_API_KEY"):
                response = generate_openai_response(prompt, filtered_data)
            else:
                response = "Please enter your OpenAI API key in the sidebar to enable AI responses."
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.chat_messages.append({"role": "assistant", "content": response})

# About section at the bottom
about_col = st.columns([6, 2])
with about_col[0]:
    with st.expander('About', expanded=True):
        st.write('''
            - This dashboard shows analytics for your travel agency's Instagram AI bot.
            - Filter by date range and platform using the sidebar.
            - The Destination Trends chart shows trends in destination interest over time.
            - Daily Booking Revenue shows revenue patterns from confirmed bookings.
            - Top Destinations and Trip Types show the most popular travel choices.
            - Event Distribution shows how users interact with your Instagram bot.
            - Use the chat interface to ask questions about the data.
            ''')
