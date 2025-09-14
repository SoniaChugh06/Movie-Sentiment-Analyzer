import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Ensure models folder exists
os.makedirs('../models', exist_ok=True)

# Load cleaned data
df = pd.read_parquet(r"C:\Users\SONIA\Movie Sentiment Analyzer\Movie-Sentiment-Analyzer\data\imdb_clean.parquet")

X = df['clean_review']
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline
pipe = make_pipeline(
    TfidfVectorizer(max_features=20000, ngram_range=(1,2)),
    LogisticRegression(max_iter=1000)
)

# Train
pipe.fit(X_train, y_train)

# Predict
preds = pipe.predict(X_test)

# Evaluation
print('Accuracy:', accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
print('Confusion Matrix:\n', confusion_matrix(y_test, preds))

# Save pipeline
joblib.dump(pipe, '../models/tfidf_logreg.joblib')
print('Saved TF-IDF + Logistic Regression pipeline to ../models/tfidf_logreg.joblib')
