
import re
import pickle
import numpy as np
import pandas as pd
import nltk

from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"

MODEL_FILE = BASE_DIR / "news_lstm_model.keras"
TOKENIZER_FILE = BASE_DIR / "tokenizer.pkl"

CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]

vocab_size = 20000
max_length = 100
embedding_dim = 100
LSTM_UNITS = 64

Train_sample_per_class = 5000

epochs = 10
batch_size = 64


nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

stop_words = set(stopwords.words("english"))
stop_words -= {"no", "nor", "not"}

lemmatizer = WordNetLemmatizer()


# text processing

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = text.split()

    processed_tokens = []
    for word in tokens:
        if word not in stop_words and len(word) > 1:
            processed_tokens.append(
                lemmatizer.lemmatize(word)
            )

    return " ".join(processed_tokens)

def build_model():
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True
        ),

        LSTM(LSTM_UNITS),

        Dropout(0.3),

        Dense(32, activation="relu"),

        Dense(len(CATEGORIES), activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# LOAD or train model

if MODEL_FILE.exists() and TOKENIZER_FILE.exists():

    print("Saved model found. Loading model...")

    model = load_model(MODEL_FILE)

    with open(TOKENIZER_FILE, "rb") as file:
        tokenizer = pickle.load(file)

    print("Model loaded successfully. Training skipped.")

else:

    print("No saved model found. Starting training...")

    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        raise FileNotFoundError(
            "Dataset files not found in the data folder."
        )

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    train_df = train_df.dropna(subset=["text", "label"]).copy()
    test_df = test_df.dropna(subset=["text", "label"]).copy()

    train_df["label"] = train_df["label"].astype(int)
    test_df["label"] = test_df["label"].astype(int)

    # training
    train_df = (
        train_df.groupby("label", group_keys=False)
        .sample(
            n=Train_sample_per_class,
            random_state=42
        )
        .reset_index(drop=True)
    )

    train_df["clean_text"] = train_df["text"].apply(clean_text)
    test_df["clean_text"] = test_df["text"].apply(clean_text)

    train_df = train_df[
        train_df["clean_text"].str.strip() != ""
    ].copy()

    test_df = test_df[
        test_df["clean_text"].str.strip() != ""
    ].copy()

    X_text = train_df["clean_text"].tolist()
    y = train_df["label"].to_numpy()

    X_test_text = test_df["clean_text"].tolist()
    y_test = test_df["label"].to_numpy()

    tokenizer = Tokenizer(
        num_words=vocab_size,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(X_text)

    X_sequences = tokenizer.texts_to_sequences(X_text)
    X_test_sequences = tokenizer.texts_to_sequences(X_test_text)

    X_padded = pad_sequences(
        X_sequences,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    X_test_padded = pad_sequences(
        X_test_sequences,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_padded,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = build_model()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(
        X_test_padded,
        y_test,
        verbose=0
    )

    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

    y_pred = np.argmax(
        model.predict(X_test_padded, verbose=0),
        axis=1
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2, 3],
            target_names=CATEGORIES,
            zero_division=0
        )
    )

    model.save(MODEL_FILE)

    with open(TOKENIZER_FILE, "wb") as file:
        pickle.dump(tokenizer, file)

    print("Model and tokenizer saved successfully.")


def predict_news(news_text):

    cleaned = clean_text(news_text)

    if not cleaned.strip():
        print("Please enter a valid news headline.")
        return

    sequence = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(
        sequence,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(padded, verbose=0)[0]

    predicted_id = int(np.argmax(prediction))
    predicted_category = CATEGORIES[predicted_id]

    confidence = float(prediction[predicted_id]) * 100

    print("\nPredicted Category:", predicted_category)
    print(f"Confidence Score: {confidence:.2f}%")

while True:

    news = input("\nEnter news headline: ").strip()

    if news.lower() == "exit":
        print("Program closed.")
        break

    if not news:
        print("Please enter a headline.")
        continue

    predict_news(news)
