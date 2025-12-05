import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="sgpai/tourism-mlops", filename="tourism_package_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism package purchase Prediction")
st.write("""
This application predicts the the likelyhood of a tourism package being purchased based on the customer and customer interaction details.
Please enter the app details below to get a purchase prediction.
""")

# User input
typeofcontact = st.selectbox("TypeofContact", ["Self Enquiry","Company Invited"])
age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
citytier = st.selectbox("CityTier", ["1","2","3"])
durationofthepitch = st.number_input("DurationOfPitch", min_value=1, max_value=100, value=10, step=1)
occupation = st.selectbox("Occupation", ["Salaried","Small Business","Large Business","Free Lancer"])
gender = st.selectbox("Gender", ["Male","Female"])
numberofpersonvisiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=10, value=1, step=1)
numberoffollowups = st.number_input("NumberOfFollowups", min_value=1, max_value=10, value=1, step=1)
productpitched = st.selectbox("ProductPitched", ["Basic","Deluxe","Standard","Super Deluxe","King"])
preferredpropertystar = st.selectbox("PreferredPropertyStar", ["3","4","5"])
maritalstatus = st.selectbox("MaritalStatus", ["Married","Divorced","Unmarried","Single"])
numberoftrips = st.number_input("NumberOfTrips", min_value=1, max_value=100, value=1, step=1)
passport = st.selectbox("Passport", ["Yes","No"])
pitchsatisfactionscore = st.selectbox("PitchSatisfactionScore", ["1","2","3","4","5"])
owncar = st.selectbox("OwnCar", ["Yes","No"])
numberofchildrenvisiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=10, value=1, step=1)
designation = st.selectbox("Designation", ["Executive","Manager","Senior Manager","AVP","VP"])
monthlyincome = st.number_input("MonthlyIncome", min_value=0, max_value=5000000, value=1000, step=100)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'TypeofContact': typeofcontact,
    'Age': age,
    'CityTier': citytier,
    'DurationOfPitch': durationofthepitch,
    'Occupation' : occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': numberofpersonvisiting,
    'NumberOfFollowups': numberoffollowups,
    'ProductPitched': productpitched,
    'PreferredPropertyStar': preferredpropertystar,
    'MaritalStatus': maritalstatus,
    'NumberOfTrips': numberoftrips,
    'Passport': 1 if passport == 'Yes' else 0,
    'PitchSatisfactionScore':pitchsatisfactionscore,
    'OwnCar': 1 if owncar == 'Yes' else 0,
    'NumberOfChildrenVisiting':numberofchildrenvisiting,
    'Designation':designation,
    'MonthlyIncome': monthlyincome
}])

# Predict button
if st.button("Predict Purchase"):
    ###prediction = 'Yes' if model.predict(input_data)[0] == 1 else 'No'
    prediction = model.predict(input_data)[0]
    st.write(f"Purchase Prediction Result: {prediction}.")
