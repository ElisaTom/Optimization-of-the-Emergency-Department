import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Page Configuration
st.set_page_config(page_title="Emergency Dept Dashboard", layout="wide")
st.title(" Emergency Department Optimization")

# Function to load data (or generate fake data if missing)
@st.cache_data
def load_data():
    try:
        # Attempt 1: Load real data
        df = pd.read_csv("data/raw/EventLog.csv")
        st.sidebar.success(" Real data loaded successfully!")


    except FileNotFoundError:
        # Attempt 2: Generate simulated data (Safety net)
        st.sidebar.warning("⚠️Real data not found. Using simulated data for demonstration.")
        
        # Create 100 fake patients
        dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
        df = pd.DataFrame({
            "Patient_ID": range(100),
            "Waiting_Time_Mins": np.random.randint(10, 120, 100), # Random mins between 10 and 120
            "START": dates
        })
        

    df['START'] = pd.to_datetime(df['START'])
    df['STOP'] = pd.to_datetime(df['STOP'])

    # Calculate Waiting Time in minutes
    df['Waiting_Time_Mins'] = (df['STOP'] - df['START']).dt.total_seconds() / 60
    df['Waiting_Time_Mins'] = df['Waiting_Time_Mins'].clip(lower=0)


    df['Arrival_Date'] = df['START'].dt.date 
    df['Arrival_Hour'] = df['START'].dt.hour

    # Weekly division
    df['Year_Week'] = df['START'].dt.strftime('%Y - Week %U')
    df['Day_Label'] = df['START'].dt.strftime('%A, %d-%m')

    return df

# Load the dataì
df = load_data()

# Visualization filters
st.sidebar.header("Control panel")
time_of_day = st.sidebar.selectbox(" Select Time of Day", ["Day (06:00 - 18:00)", "Night (18:00 - 06:00)"])
if time_of_day == "Day (06:00 - 18:00)":
    df_time = df[(df['Arrival_Hour'] >= 6) & (df['Arrival_Hour'] < 18)]
elif time_of_day == "Night (18:00 - 06:00)":
    df_time = df[(df['Arrival_Hour'] >= 18) | (df['Arrival_Hour'] < 6)]
weeks = sorted(df_time['Year_Week'].unique())
selected_weeks = st.sidebar.selectbox(" Select Weeks", weeks)

df_time = df[df['Year_Week'] == selected_weeks]

# Key Metrics
st.metric("Total Patients Processed", len(df_time))
st.write("---")
st.caption(" Dashboard running inside a Docker Container.")

# --- DASHBOARD LAYOUT ---
st.subheader("📈 Waiting times per Week")
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df_time['Day_Name'] = df_time['START'].dt.strftime('%A')

data_plot = (
    df_time.groupby(['Arrival_Date', 'Day_Label', 'Day_Name'])['Waiting_Time_Mins']
    .mean()
    .reset_index()
    .sort_values('Arrival_Date')
)

data_plot['Day_Name'] = pd.Categorical(data_plot['Day_Name'], categories=day_order, ordered=True)
data_plot = data_plot.sort_values('Day_Name')

fig1 = px.bar(
data_plot, 
x='Day_Label',
y='Waiting_Time_Mins',
text=data_plot['Waiting_Time_Mins'].round(1),
labels={'Waiting_Time_Mins': 'Avg Wait (min)', 'Day_Label': 'Day'}
)
    
fig1.update_layout(
xaxis_type="category",
xaxis_title ='Day of the Week', 
yaxis_title="Waiting time in minutes",
)

st.plotly_chart(fig1, use_container_width=True)


st.subheader(" Waiting Time Distribution")
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
title = 'Average Waiting Time by Hour',
labels={'Waiting_Time_Mins': 'Average Waiting Time (Mins)', 'Arrival_Hour': 'Hour of Day'}
    )  
fig2.add_hline(y=60, line_dash="dash", line_color="gray", annotation_text="60 min limit")
st.plotly_chart(fig2, use_container_width=True)
