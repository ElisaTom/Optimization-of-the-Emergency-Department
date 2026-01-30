import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Page Configuration
st.set_page_config(page_title="Emergency Dept Dashboard", layout="wide")
st.title("Emergency Department Optimization")

# Function to load data (or generate simulated data if the file is inaccessible)
@st.cache_data
def load_data():
    try:
        # Load the CSV file using the correct separator
        df = pd.read_csv("data/raw/EventLog.csv", sep=";")
        st.sidebar.success("Real data loaded successfully.")

    except FileNotFoundError:
        # Fallback: Generate simulated data for demonstration purposes
        st.sidebar.warning("Real data not found. Using simulated data for demonstration.")
        
        # Create 100 simulated patient records
        dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
        df = pd.DataFrame({
            "Patient_ID": range(100),
            "Waiting_Time_Mins": np.random.randint(10, 120, 100), 
            "START": dates,
            "STOP": dates + pd.to_timedelta(np.random.randint(10, 120, 100), unit='m')
        })
    
    # Data Cleaning: Remove potential whitespace from column names
    df.columns = df.columns.str.strip()

    # Datetime Conversion with error coercion
    df['START'] = pd.to_datetime(df['START'], utc=True, errors='coerce')
    df['STOP'] = pd.to_datetime(df['STOP'], utc=True, errors='coerce')

    # Remove rows with invalid timestamps
    df = df.dropna(subset=['START', 'STOP'])

    # Calculate Waiting Time in minutes
    df['Waiting_Time_Mins'] = (df['STOP'] - df['START']).dt.total_seconds() / 60
    # Ensure no negative values exist
    df['Waiting_Time_Mins'] = df['Waiting_Time_Mins'].clip(lower=0)

    # Derive additional temporal features
    df['Arrival_Date'] = df['START'].dt.date 
    df['Arrival_Hour'] = df['START'].dt.hour
    df['Year_Week'] = df['START'].dt.strftime('%Y - Week %U')
    df['Day_Label'] = df['START'].dt.strftime('%A, %d-%m')

    return df

# Load the dataset
df = load_data()

# Termination check if data loading failed completely
if df.empty:
    st.error("The dataset is empty or the date format could not be parsed.")
    st.stop()

# --- CONTROL PANEL ---
st.sidebar.header("Control Panel")

# Time of Day Filter
time_of_day = st.sidebar.selectbox("Select Time of Day", ["All Day", "Day (06:00 - 18:00)", "Night (18:00 - 06:00)"])

# Filter logic based on time of day
if time_of_day == "Day (06:00 - 18:00)":
    df_time = df[(df['Arrival_Hour'] >= 6) & (df['Arrival_Hour'] < 18)]
elif time_of_day == "Night (18:00 - 06:00)":
    df_time = df[(df['Arrival_Hour'] >= 18) | (df['Arrival_Hour'] < 6)]
else:
    df_time = df.copy()

# Week Selection Filter (UPDATED LOGIC)
if not df_time.empty:
    # Create a list that includes "All Weeks" at the top
    unique_weeks = sorted(df_time['Year_Week'].unique())
    week_options = ["All Weeks"] + unique_weeks
    
    selected_week = st.sidebar.selectbox("Select Weeks", week_options)
    
    # Apply filter only if a specific week is selected
    if selected_week != "All Weeks":
        df_time = df_time[df_time['Year_Week'] == selected_week]

# Display Key Metrics
st.metric("Total Patients Processed", len(df_time))
st.write("---")
st.caption("Dashboard running locally.")

# --- DASHBOARD VISUALIZATIONS ---
st.subheader("Waiting Times Overview")

if not df_time.empty:
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_time['Day_Name'] = df_time['START'].dt.strftime('%A')

    # Aggregation for the first chart
    # If viewing "All Weeks", we group by Day Name to see general trends
    if selected_week == "All Weeks":
         data_plot = (
            df_time.groupby(['Day_Name'])['Waiting_Time_Mins']
            .mean()
            .reset_index()
        )
         x_axis = 'Day_Name'
         title_text = "Average Waiting Time by Day of Week (Global)"
    else:
        # If viewing single week, keep the detailed view
        data_plot = (
            df_time.groupby(['Day_Label', 'Day_Name'])['Waiting_Time_Mins']
            .mean()
            .reset_index()
            .sort_values('Day_Label')
        )
        x_axis = 'Day_Label'
        title_text = "Average Waiting Time by Day (Selected Week)"

    data_plot['Day_Name'] = pd.Categorical(data_plot['Day_Name'], categories=day_order, ordered=True)
    data_plot = data_plot.sort_values('Day_Name')

    fig1 = px.bar(
        data_plot, 
        x=x_axis,
        y='Waiting_Time_Mins',
        text=data_plot['Waiting_Time_Mins'].round(1),
        labels={'Waiting_Time_Mins': 'Avg Wait (min)', x_axis: 'Day'},
        title=title_text
    )
        
    fig1.update_layout(yaxis_title="Waiting Time (minutes)")
    
    # FIX: Updated syntax to silence warnings (removed use_container_width)
    st.plotly_chart(fig1)

    # Second Chart: Hourly Distribution
    st.subheader("Hourly Performance")
    avg_by_hour = (
        df_time
        .groupby("Arrival_Hour")["Waiting_Time_Mins"]
        .mean()
        .reset_index()
    )

    fig2 = px.bar(
        avg_by_hour,
        x= "Arrival_Hour",
        y= "Waiting_Time_Mins",
        title = 'Average Waiting Time by Hour of Day',
        labels={'Waiting_Time_Mins': 'Average Waiting Time (Mins)', 'Arrival_Hour': 'Hour of Day'}
    )  
    fig2.add_hline(y=60, line_dash="dash", line_color="red", annotation_text="Target Limit (60m)")
    
    # FIX: Updated syntax to silence warnings
    st.plotly_chart(fig2)

else:
    st.info("No data available for the selected filters.")