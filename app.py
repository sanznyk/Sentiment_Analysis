import streamlit as st
import pickle

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Streamlit app
st.title("Amazon Review Sentiment Predictor")

# Text input
user_input = st.text_area("Enter a review:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform input text using the same vectorizer used during training
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]

        # You can customize label interpretation based on your model's classes
        label = "Positive" if prediction == 1 else "Negative"
        st.success(f"Predicted Sentiment: {label}")
