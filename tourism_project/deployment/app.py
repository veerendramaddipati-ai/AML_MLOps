
import streamlit as st
import pandas as pd
import joblib
import os
import traceback


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Updated to load the entire pipeline
MODEL_PATH = os.path.join(
    CURRENT_DIR,
    "best_model.pkl"
)

# Initialize model to None
model = None

try:
    # Load the entire pipeline (preprocessor + model)
    model = joblib.load(MODEL_PATH)
    st.success("Model pipeline loaded successfully")

except Exception as e:
    st.error(f"Error loading model pipeline: {str(e)}")
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

# Adding input fields for 'TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 'ProductPitched', 'Designation' for preprocessing
type_of_contact = st.selectbox("Type of Contact", ['Company Invited', 'Self Inquiry'])
occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Freelancer', 'Large Business', 'Unemployed'])
gender = st.selectbox("Gender", ['Male', 'Female'])
marital_status = st.selectbox("Marital Status", ['Married', 'Single', 'Divorced'])
product_pitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'Executive', 'Luxury'])
designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP', 'Director'])

# Adding fields for NumberOfPersonVisiting, PreferredPropertyStar, NumberOfTrips, NumberOfChildrenVisiting
number_of_person_visiting = st.number_input("Number of Persons Visiting", 1, 10, 1)
preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
number_of_trips = st.number_input("Number of Trips Annually", 1, 100, 1)
number_of_children_visiting = st.number_input("Number of Children Visiting", 0, 5, 0)
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", 1, 5, 3)



if st.button("Predict"):

    input_data_raw = {
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": number_of_person_visiting,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": number_of_trips,
        "Passport": passport,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": number_of_children_visiting,
        "Designation": designation,
        "MonthlyIncome": income,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "ProductPitched": product_pitched,
        "NumberOfFollowups": followups,
        "DurationOfPitch": duration
    }

    input_df = pd.DataFrame([input_data_raw])

    # The loaded model is now the full pipeline, which handles preprocessing internally
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success(
            "Customer is Likely to Purchase"
        )
    else:
        st.error(
            "Customer is Unlikely to Purchase"
        )
