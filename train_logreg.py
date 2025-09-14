import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
import os

# Load data
df = pd.read_parquet(r"C:\Users\SONIA\Movie Sentiment Analyzer\Movie-Sentiment-Analyzer\data\imdb_clean.parquet")
texts = df['review'].astype(str)
labels = df['label'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create pipeline
pipe = Pipeline([
    ('tfidfvectorizer', TfidfVectorizer(max_features=10000)),
    ('logisticregression', LogisticRegression(max_iter=500))
])

# Train model
pipe.fit(X_train, y_train)

# Ensure models folder exists
os.makedirs(r"C:\Users\SONIA\Movie Sentiment Analyzer\Movie-Sentiment-Analyzer\models", exist_ok=True)

# Save pipeline
joblib.dump(pipe, r"C:\Users\SONIA\Movie Sentiment Analyzer\Movie-Sentiment-Analyzer\models\tfidf_logreg.joblib")

print("TF-IDF + Logistic Regression pipeline created and saved successfully!")
