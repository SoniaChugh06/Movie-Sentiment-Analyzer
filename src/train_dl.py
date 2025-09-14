import json
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import os

# ------------------ CONFIG ------------------ #
NUM_WORDS = 20000      # vocabulary size
MAXLEN = 200           # max sequence length
EMBED_DIM = 128        # embedding dimension
BATCH_SIZE = 128
EPOCHS = 8

# ------------------ LOAD DATA ------------------ #
data_path = r"C:\Users\SONIA\Movie Sentiment Analyzer\Movie-Sentiment-Analyzer\data\imdb_clean.parquet"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"{data_path} not found!")

df = pd.read_parquet(data_path)
texts = df['review'].astype(str).fillna('').tolist()  # ensure no NaNs
labels = df['label'].values

# ------------------ SPLIT DATA ------------------ #
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# ------------------ TOKENIZER ------------------ #
tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train_texts[:5000])  # test with first 5000 for speed

X_train_seq = tokenizer.texts_to_sequences(X_train_texts)
X_train_pad = pad_sequences(X_train_seq, maxlen=MAXLEN, padding='post', truncating='post')

X_test_seq = tokenizer.texts_to_sequences(X_test_texts)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAXLEN, padding='post', truncating='post')

# ------------------ MODEL ------------------ #
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(NUM_WORDS, EMBED_DIM),  # input_length is optional
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ------------------ CALLBACKS ------------------ #
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# ------------------ TRAIN ------------------ #
history = model.fit(
    X_train_pad, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es]
)

# ------------------ EVALUATE ------------------ #
loss, acc = model.evaluate(X_test_pad, y_test)
print('Test accuracy:', acc)

# ------------------ SAVE MODEL & TOKENIZER ------------------ #
model_dir = r"C:\Users\SONIA\Movie Sentiment Analyzer\Movie-Sentiment-Analyzer\models"
os.makedirs(model_dir, exist_ok=True)

model.save(os.path.join(model_dir, 'lstm_model.keras'))  # use Keras format
with open(os.path.join(model_dir, 'tokenizer.json'), 'w') as f:
    f.write(tokenizer.to_json())

print('Saved LSTM model and tokenizer in', model_dir)
