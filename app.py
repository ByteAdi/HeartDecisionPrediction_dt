
import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('svc_model.joblib')

# Title and description
st.title("Heart Disease Prediction")
st.write("This app predicts the likelihood of heart disease using an SVM model.")

# Define feature input form
st.sidebar.header("User Input Features")
def user_input_features():
    age = st.sidebar.slider("Age", 29, 77, 50)
    sex = st.sidebar.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
    cp = st.sidebar.slider("Chest Pain Type (0-3)", 0, 3, 1)
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 94, 200, 120)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 126, 564, 250)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [1, 0])
    restecg = st.sidebar.slider("Resting ECG Results (0-2)", 0, 2, 1)
    thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 71, 202, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", 0.0, 6.2, 1.0)
    slope = st.sidebar.slider("Slope of the Peak Exercise ST Segment (0-2)", 0, 2, 1)
    ca = st.sidebar.slider("Number of Major Vessels Colored by Fluoroscopy (0-4)", 0, 4, 0)
    thal = st.sidebar.slider("Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)", 0, 2, 1)
    
    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    features = np.array(list(data.values())).reshape(1, -1)
    return features

# Input features from user
input_features = user_input_features()

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_features)
    st.subheader("Prediction")
    if prediction[0] == 1:
        st.write("The patient is likely to have heart disease.")
    else:
        st.write("The patient is unlikely to have heart disease.")
