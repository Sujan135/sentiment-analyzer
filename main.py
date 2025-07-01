import re
from data.dataset import load_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from utils.metrics import show_metrics
from collections import Counter
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = text.strip()
    return text

def tune_nb(X_train, y_train):
    nb = MultinomialNB()
    params = {'alpha': [0.1, 0.5, 1.0, 5.0, 10.0]}
    grid = GridSearchCV(nb, params, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    print("Best alpha found:", grid.best_params_['alpha'])
    return grid.best_estimator_

def main():
    data = load_data()
    texts = [clean_text(x[0]) for x in data]
    raw_labels = [x[1] for x in data]
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(raw_labels)

    print("Class distribution:", Counter(raw_labels))

    stop_words = stopwords.words('english')
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=stop_words)
    X = vectorizer.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.3, random_state=42, stratify=labels
    )

    model = tune_nb(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Actual labels:", label_encoder.inverse_transform(y_test))
    print("Predicted labels:", label_encoder.inverse_transform(y_pred))
    show_metrics(label_encoder.inverse_transform(y_test), label_encoder.inverse_transform(y_pred))

    model.fit(X, labels)
    y_pred_full = model.predict(X)
    full_acc = accuracy_score(labels, y_pred_full)
    print("Full data accuracy:", full_acc)
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
