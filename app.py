import streamlit as st
import pickle

# Load trained model and vectorizer
with open('fake_news_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    import string
    from nltk.corpus import stopwords
    import nltk
    nltk.download('stopwords', quiet=True)
    stop = set(stopwords.words('english'))
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop]
    return ' '.join(words)

st.title("Fake News Detection App")
user_input = st.text_area("Paste your news article or headline:")

if st.button("Check if Fake or Real"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]
        label = "REAL News" if prediction == 1 else "FAKE News"
        st.success(f"Prediction: {label}")
