import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample keywords for success and failure
success_keywords = ['innovative', 'cutting-edge', 'market leader', 'high growth', 'strong customer base',
    # (your success keywords)
]

failure_keywords = ['struggling', 'financial difficulties', 'poor management', 'bankruptcy', 'limited resources',
    # (your failure keywords)
]

# Generate synthetic dataset
data = {
    'description': [
        'innovative product with cutting-edge technology',
        'struggling to gain market traction',
        'successful startup with a strong customer base',
        'facing financial difficulties',
        'disruptive solution for industry challenges',
        'poor management and lack of direction',
        'rapid growth and expansion',
        'declining user engagement',
        'revolutionary idea but limited resources',
        'established market leader with consistent growth',
        'failing to collaborate with vendors'
    ] * 50,  # Repeat the descriptions to create a larger dataset
    'success_label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0] * 50  # 0: Failure, 1: Success
}

df = pd.DataFrame(data)

# Streamlit app
st.title("Startup Success Prediction")

# User input for startup features
user_input = st.text_area("Enter a description of your startup:")

if st.button("Predict"):
    # Convert user input to numerical features using TF-IDF
    user_features = pd.DataFrame({
        'success_feature': [any(keyword in user_input for keyword in success_keywords)],
        'failure_feature': [any(keyword in user_input for keyword in failure_keywords)]
    })

    print("Column names in DataFrame:", df.columns)

    # Assuming 'success_label' is the target variable
    features = df[['success_feature', 'failure_feature']]
    target = df['success_label']

    # Initialize a RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(features, target)

    # Make prediction for user input
    prediction = model.predict(user_features)

    # Display prediction
    if prediction[0] == 1:
        st.success("The model predicts that your startup is likely to succeed. Good luck! for a start to break bounds of everyone's imagination.")
    else:
        st.error("The model predicts that your startup is likely to fail. Consider providing more appealing features to customers and take up the universe.")
