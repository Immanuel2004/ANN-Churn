import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle

@st.cache_resource
def load_model_and_encoders():
    model = tf.keras.models.load_model('Model.h5')

    with open('Label_Encoder_Gender.pkl','rb') as file:
        gender_encoder = pickle.load(file)

    with open('OHE_Geography.pkl','rb') as file:
        geo_encoder = pickle.load(file)

    with open('Scaler.pkl','rb') as file:
        scaler = pickle.load(file)

    return model, gender_encoder, geo_encoder, scaler

Model, Encoded_Gender, Encoded_Geography, Scaler = load_model_and_encoders()

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("Customer Churn Prediction")
st.markdown("Estimate the probability of a customer churning based on their profile.")

st.header("Customer Information")

geography = st.selectbox('Geography', Encoded_Geography.categories_[0])
gender = st.selectbox('Gender', Encoded_Gender.classes_)
age = st.slider('Age', 18, 90, 30)

credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=650)
balance = st.number_input('Balance', min_value=0.0, step=100.0)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, step=500.0)

tenure = st.slider('Tenure (Years)', 0, 10, 5)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_cr_card = st.selectbox('Has Credit Card?', [0, 1])
is_active_member = st.selectbox('Active Member?', [0, 1])


def preprocess_and_predict():
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [Encoded_Gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_encoded = Encoded_Geography.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=Encoded_Geography.get_feature_names_out(['Geography'])
    )

    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    input_data = input_data.reindex(columns=Scaler.feature_names_in_, fill_value=0)

    input_scaled = Scaler.transform(input_data)

    prediction = Model.predict(input_scaled)
    return prediction[0][0]


if st.button("Predict"):
    prediction_proba = preprocess_and_predict()
    churn_percentage = prediction_proba * 100

    st.subheader("Prediction Result")
    st.write(f"Churn Probability: **{churn_percentage:.2f}%**")

    st.progress(int(churn_percentage))

    if prediction_proba > 0.5:
        st.error("The customer is likely to churn.")
    else:
        st.success("The customer is not likely to churn.")
