import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor  # Replace with your actual ML model
import random

# Title and Introduction
st.title("AI-Powered Water Demand Prediction")
st.write("""
This application uses Artificial Intelligence to predict water demand based on environmental, demographic, and industrial factors. Adjust the parameters below to see how demand changes and explore insights from AI predictions.
""")

# Sidebar for User Inputs
st.sidebar.header("Input Parameters")
temperature = st.sidebar.slider("Temperature (°C)", min_value=10, max_value=50, value=25)
precipitation = st.sidebar.slider("Precipitation (mm)", min_value=0, max_value=200, value=50)
population = st.sidebar.number_input("Population (thousands)", min_value=1, max_value=10000, value=500)
industrial_usage = st.sidebar.slider("Industrial Usage (%)", min_value=0, max_value=100, value=30)
season = st.sidebar.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"])

# Combine inputs into a feature set
features = {
    "Temperature (°C)": temperature,
    "Precipitation (mm)": precipitation,
    "Population (thousands)": population,
    "Industrial Usage (%)": industrial_usage,
    "Season": season,
}
features_df = pd.DataFrame([features])

st.write("### Input Parameters")
st.dataframe(features_df)

# Dummy prediction function (replace with your AI model)
def predict_water_demand(features):
    # Simulate predictions with random values (Replace with your model)
    prediction = random.uniform(1000, 5000)  # Dummy prediction in liters/day
    return prediction

# Prediction and Visualization
if st.button("Predict Water Demand"):
    prediction = predict_water_demand(features_df)
    st.write(f"### Predicted Water Demand: {prediction:.2f} liters/day")
    st.write("The prediction is powered by an AI model considering the provided input parameters.")

    # Generate example charts
    st.write("### Visualization of Historical Trends")
    days = list(range(1, 31))
    historical_demand = np.random.uniform(3000, 7000, 30)
    historical_supply = np.random.uniform(3000, 7000, 30)
    plt.figure(figsize=(10, 5))
    plt.plot(days, historical_demand, label="Water Demand", color="blue")
    plt.plot(days, historical_supply, label="Water Supply", color="green")
    plt.xlabel("Days")
    plt.ylabel("Liters")
    plt.title("Historical Water Demand and Supply Trends")
    plt.legend()
    st.pyplot(plt)

# Footer
st.write("""
---
**Note:** This is a prototype interface. Integrate your trained AI model to replace the dummy prediction logic.
""")
