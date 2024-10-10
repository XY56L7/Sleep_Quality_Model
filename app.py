import streamlit as st
import pandas as pd
import joblib  # For loading the saved model
import matplotlib.pyplot as plt
import numpy as np

# Load the trained model
model = joblib.load('model.pkl')

# Load the CSV file for statistics
data = pd.read_csv('Health_Sleep_Statistics.csv')

# Streamlit app layout with tabs
tab1, tab2 = st.tabs(["Sleep Quality Prediction", "Health Statistics"])

with tab1:
    st.title("Sleep Quality Predictor")

    # Input fields
    age = st.number_input("Age", min_value=0, max_value=100)
    gender = st.selectbox("Gender", ("male", "female"))
    daily_steps = st.number_input("Daily Steps", min_value=0)
    calories_burned = st.number_input("Calories Burned", min_value=0)
    physical_activity_level = st.selectbox("Physical Activity Level", ("low", "medium", "high"))
    dietary_habits = st.selectbox("Dietary Habits", ("unhealthy", "medium", "healthy"))
    sleep_disorders = st.selectbox("Sleep Disorders", ("yes", "no"))

    # Encoding inputs like in the trained model
    gender_m = 1 if gender == "male" else 0
    physical_activity_low = 1 if physical_activity_level == "low" else 0
    physical_activity_medium = 1 if physical_activity_level == "medium" else 0
    dietary_medium = 1 if dietary_habits == "medium" else 0
    dietary_unhealthy = 1 if dietary_habits == "unhealthy" else 0
    sleep_disorders_yes = 1 if sleep_disorders == "yes" else 0

    # Create the input data for prediction
    input_data = pd.DataFrame({
        'Age': [age],
        'Daily Steps': [daily_steps],
        'Calories Burned': [calories_burned],
        'Gender_m': [gender_m],
        'Physical Activity Level_low': [physical_activity_low],
        'Physical Activity Level_medium': [physical_activity_medium],
        'Dietary Habits_medium': [dietary_medium],
        'Dietary Habits_unhealthy': [dietary_unhealthy],
        'Sleep Disorders_yes': [sleep_disorders_yes]
    })

    # Make the prediction
    if st.button("Predict Sleep Quality"):
        prediction = model.predict(input_data)
        prediction_percent = abs(prediction[0] * 100)
        st.write(f"Predicted Sleep Quality: {prediction_percent:.2f}%")

with tab2:
    st.title("Health and Sleep Statistics")

    # Visualization of Age vs Sleep Quality
    st.subheader("Age vs Sleep Quality")
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Age'], data['Sleep Quality'], alpha=0.5)

    # Add regression line
    slope, intercept = np.polyfit(data['Age'], data['Sleep Quality'], 1)
    plt.plot(data['Age'], slope * data['Age'] + intercept, color='red', label='Regression Line')

    plt.title('Age vs Sleep Quality')
    plt.xlabel('Age')
    plt.ylabel('Sleep Quality (%)')
    plt.legend()
    st.pyplot(plt)

    # Visualization of Daily Steps vs Sleep Quality
    st.subheader("Daily Steps vs Sleep Quality")
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Daily Steps'], data['Sleep Quality'], color='orange', alpha=0.5)

    # Add regression line
    slope, intercept = np.polyfit(data['Daily Steps'], data['Sleep Quality'], 1)
    plt.plot(data['Daily Steps'], slope * data['Daily Steps'] + intercept, color='red', label='Regression Line')

    plt.title('Daily Steps vs Sleep Quality')
    plt.xlabel('Daily Steps')
    plt.ylabel('Sleep Quality (%)')
    plt.legend()
    st.pyplot(plt)
