
import streamlit as st
import pandas as pd
import joblib
import os
import joblib
import traceback


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    CURRENT_DIR,
    "best_model.pkl"
)


try:
    model = joblib.load(MODEL_PATH)
    st.success("Model loaded successfully")

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.code(traceback.format_exc())
    st.stop()

st.title("Wellness Tourism Package Prediction")

age = st.number_input("Age",18,80,30)
city = st.selectbox("City Tier",[1,2,3])
income = st.number_input("Monthly Income",10000,500000,30000)

passport = st.selectbox(
    "Passport",
    [0,1]
)

own_car = st.selectbox(
    "Own Car",
    [0,1]
)

followups = st.number_input(
    "Number Of Followups",
    0,
    10,
    2
)

duration = st.number_input(
    "Duration Of Pitch",
    1,
    120,
    10
)

if st.button("Predict"):

    input_df = pd.DataFrame(
        {
            "Age":[age],
            "CityTier":[city],
            "MonthlyIncome":[income],
            "Passport":[passport],
            "OwnCar":[own_car],
            "NumberOfFollowups":[followups],
            "DurationOfPitch":[duration]
        }
    )

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success(
            "Customer is Likely to Purchase"
        )
    else:
        st.error(
            "Customer is Unlikely to Purchase"
        )
