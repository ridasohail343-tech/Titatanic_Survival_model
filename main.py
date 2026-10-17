import pickle
import pandas as pd
import streamlit as st

# Load the saved model and preprocessor
with open("titanic_model.pkl", "rb") as file:
    saved_data = pickle.load(file)

model = saved_data["model"]
preprocessor = saved_data["preprocessor"]

# Streamlit page configuration
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

st.title("Titanic Survival Prediction")
st.write("Enter passenger details to predict the likelihood of survival.")

# User input
pclass = st.selectbox(
    "Passenger Class",
    options=[1, 2, 3]
)

sex = st.selectbox(
    "Gender",
    options=["male", "female"]
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=25.0,
    step=1.0
)

sibsp = st.number_input(
    "Number of Siblings/Spouses Aboard",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

parch = st.number_input(
    "Number of Parents/Children Aboard",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

fare = st.number_input(
    "Ticket Fare",
    min_value=0.0,
    value=32.0,
    step=1.0
)

embarked = st.selectbox(
    "Port of Embarkation",
    options=["S", "C", "Q"],
    format_func=lambda x: {
        "S": "Southampton",
        "C": "Cherbourg",
        "Q": "Queenstown"
    }[x]
)

# Prediction
if st.button("Predict Survival"):
    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    })

    # Apply the same preprocessing used during training
    processed_data = preprocessor.transform(input_data)

    prediction = model.predict(processed_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("The model predicts that the passenger survived.")
    else:
        st.error("The model predicts that the passenger did not survive.")

    # Display predicted survival probability when supported
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(processed_data)[0]
        st.write(
            f"Estimated survival probability: {probabilities[1] * 100:.2f}%"
        )
