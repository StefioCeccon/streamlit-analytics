import streamlit as st
import numpy as np
import pandas as pd
import time
import os
from sqlalchemy import create_engine

st.markdown("# Main page 🎈")
st.sidebar.markdown("# Main page 🎈")

st.image("static/image.png")

## set up db

# Get database credentials from environment variables
# username = os.environ.get("POSTGRES_USERNAME")
# password = os.environ.get("POSTGRES_PASSWORD")
# host = os.environ.get("POSTGRES_HOST")
# port = os.environ.get("POSTGRES_PORT")
# database = os.environ.get("POSTGRES_DATABASE")

username = 'doadmin'
password = 'eifut7zbwv9a5akc'
host = 'db-postgresql-skipq-logger-1-do-user-8711821-0.b.db.ondigitalocean.com'
port = '25060'
database = 'defaultdb'


# Create connection string
connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"

# Create database connection
@st.cache_resource
def get_connection():
    return create_engine(connection_string)

## set up page

'Starting a long computation...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.005)

'...and now we\'re done!'

conn = get_connection()
# Replace conn.query with proper SQLAlchemy execution
df = pd.read_sql("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';", conn)
st.dataframe(df)

if "counter" not in st.session_state:
    st.session_state.counter = 0

st.session_state.counter += 1

st.header(f"This page has run {st.session_state.counter} times.")
st.button("Run it again")

# hook the df to the session state so that it persists across runs, so everytime you click it doesnt change/reload!
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(np.random.randn(20, 2), columns=["x", "y"])

st.header("Choose a datapoint color")
color = st.color_picker("Color", "#FF0000")
st.divider()
st.scatter_chart(st.session_state.df, x="x", y="y", color=color)

st.text_input("Your name", key="name")

# You can access the value at any point with:
st.session_state.name

left_column, right_column = st.columns(2)
# You can use a column just like st.sidebar:
left_column.button('Press me!')

# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")

if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    chart_data


df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
    })

with left_column:
    option = st.selectbox(
        'Which number do you like best?',
        df['first column'])

'You selected: ', option

# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
)

dataframe = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))

st.dataframe(dataframe.style.highlight_max(axis=0))

import streamlit as st
x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)


chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)