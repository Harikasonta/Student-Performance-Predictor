import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("student_data.csv")

X = data.drop("final_score", axis=1)
y = data["final_score"]

# Train model
model = LinearRegression()
model.fit(X, y)

st.title("Student Performance Predictor")

study_hours = st.number_input("Study Hours", min_value=0)
attendance = st.number_input("Attendance (%)", min_value=0)
sleep_hours = st.number_input("Sleep Hours", min_value=0)
previous_score = st.number_input("Previous Score", min_value=0)
stress_level = st.number_input("Stress Level (1-3)", min_value=1, max_value=3)

if st.button("Predict"):
    new_student = pd.DataFrame(
        [[study_hours, attendance, sleep_hours, previous_score, stress_level]],
        columns=X.columns
    )
    prediction = model.predict(new_student)
    st.success(f"Predicted Final Score: {int(prediction[0])}")