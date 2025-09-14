import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load ML model + vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("logistic_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

# Load DL model + tokenizer
lstm_model = load_model("lstm_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAXLEN = 200

st.title("🎬 Movie Sentiment Analyzer")

review = st.text_area("Enter a movie review:")

if st.button("Analyze"):
    if review.strip() == "":
        st.warning("Please enter a review first!")
    else:
        # --- ML Prediction (Logistic Regression) ---
        X_tfidf = vectorizer.transform([review])
        ml_prob = lr_model.predict_proba(X_tfidf)[0][1]  # probability positive

        # --- DL Prediction (LSTM) ---
        seq = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(seq, maxlen=MAXLEN)
        dl_prob = float(lstm_model.predict(padded, verbose=0)[0][0])

        # You said you want just one number:
        final_prob = (ml_prob + dl_prob) / 2  # average of both

        st.metric("Sentiment Score", f"{final_prob:.2f}")
