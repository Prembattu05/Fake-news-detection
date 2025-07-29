import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop = set(stopwords.words('english'))

# Define text preprocessing — should match your training procedure!
def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop]
    return ' '.join(words)

# Load model and vectorizer
with open('fake_news_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

st.title("Fake News Detector")
user_input = st.text_area("Paste a news article (or headline):", height=200)

if st.button("Check if Fake or Real"):
    if user_input.strip() == "":
        st.warning("Please enter text to test.")
    else:
        cleaned = clean_text(user_input)
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]
        label = "REAL News" if prediction == 1 else "FAKE News"
        st.success(f"Prediction: {label}")
