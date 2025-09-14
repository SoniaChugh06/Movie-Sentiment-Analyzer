import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (only needs to run once)
nltk.download('stopwords')
nltk.download('wordnet')

stopwords_set = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text, remove_stopwords=True, lemmatize=True):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)   # remove HTML
    text = re.sub(r"[^a-z\s]", " ", text)  # keep only letters and spaces
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stopwords_set]
    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def load_and_clean(csv_path, sample_frac=None):
    df = pd.read_csv(csv_path)
    if sample_frac is not None:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
    df['clean_review'] = df['review'].apply(lambda t: clean_text(t))
    # Map sentiment to 0/1
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return df

if __name__ == '__main__':
    df = load_and_clean(r"C:\Users\SONIA\Movie Sentiment Analyzer\Movie-Sentiment-Analyzer\data\IMDB Dataset.csv",sample_frac=0.1)
    df.to_parquet(r"C:\Users\SONIA\Movie Sentiment Analyzer\Movie-Sentiment-Analyzer\data\imdb_clean.parquet", index=False)
    print('Saved cleaned dataset to data/imdb_clean.parquet')
