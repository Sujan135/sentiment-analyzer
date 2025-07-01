import re
from data.dataset import load_data
from models.naive_bayes import NaiveBayesModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from utils.metrics import show_metrics
from collections import Counter
from sklearn.preprocessing import LabelEncoder

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # Remove non-word chars
    text = text.strip()
    return text

def main():
    data = load_data()
    texts = [clean_text(x[0]) for x in data]
    raw_labels = [x[1] for x in data]
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(raw_labels)

    print("Class distribution:", Counter(raw_labels))

    vectorizer = TfidfVectorizer(ngram_range=(1, 1))  # unigrams only
    X = vectorizer.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.3, random_state=42, stratify=labels
    )

    model = NaiveBayesModel()
    model.train(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Actual labels:", label_encoder.inverse_transform(y_test))
    print("Predicted labels:", label_encoder.inverse_transform(y_pred))
    show_metrics(label_encoder.inverse_transform(y_test), label_encoder.inverse_transform(y_pred))

    model.train(X, labels)
    y_pred_full = model.predict(X)
    print("Full data accuracy:", (y_pred_full == labels).mean())
    show_metrics(label_encoder.inverse_transform(labels), label_encoder.inverse_transform(y_pred_full))

    while True:
        user_input = input("\nEnter a review (or type 'exit'): ")
        if user_input.lower() == 'exit':
            break
        clean_input = clean_text(user_input)
        vec = vectorizer.transform([clean_input])
        pred_numeric = model.predict(vec)[0]
        result = label_encoder.inverse_transform([pred_numeric])[0]
        print("Prediction:", result)

if __name__ == "__main__":
    main()
