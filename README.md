# Fake-news-detection
This  is a project based on fake news detection in which the algorithm analyses the matter and define whether ir is fake or real

# Fake News Detection

A machine learning project to classify news articles as **fake** or **real** using Natural Language Processing techniques.

## Features
- Cleans and processes raw news text
- Uses TF-IDF vectorization
- Trains a classifier (Logistic Regression by default)
- Evaluates performance (accuracy, precision, recall, F1)
- Optional: Simple web app interface with Streamlit

## How to Use

1. **Setup:**  
   Install required libraries:
   ```
   pip install pandas numpy scikit-learn nltk matplotlib seaborn streamlit
   ```
2. **Data:**  
   Place `Fake.csv` and `True.csv` datasets in the project folder.

3. **Notebook:**  
   Open the Jupyter/Colab notebook (`.ipynb`) and run step-by-step to train and test the model.

4. **Streamlit App (optional):**  
   Make sure you have:
   - `fake_news_model.pkl`
   - `tfidf_vectorizer.pkl`
   - `app.py`
   Run:
   ```
   streamlit run app.py
   ```

## Project Files
- `fake_news_detection.ipynb` — Main notebook
- `app.py` — Streamlit app
- Model/data files (`.csv`, `.pkl`)

---

*Author**: Prem Battu
