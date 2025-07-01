import re
import joblib
from data.dataset import load_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from utils.metrics import show_metrics
from collections import Counter
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    tokens = text.split()
    stemmed = [ps.stem(word) for word in tokens]
    return ' '.join(stemmed)

def tune_naive_bayes(X_train, y_train):
    nb = MultinomialNB()
    params = {'alpha': [0.1, 0.5, 1.0, 5.0]}
    grid = GridSearchCV(nb, params, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    print("Best Naive Bayes alpha:", grid.best_params_['alpha'])
    return grid.best_estimator_

def tune_logistic_regression(X_train, y_train):
    lr = LogisticRegression(solver='liblinear', max_iter=1000)
    params = {'C': [0.01, 0.1, 1, 10]}
    grid = GridSearchCV(lr, params, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    print("Best Logistic Regression C:", grid.best_params_['C'])
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

    nb_model = tune_naive_bayes(X_train, y_train)
    lr_model = tune_logistic_regression(X_train, y_train)

    print("\n--- Naive Bayes Evaluation ---")
    nb_pred = nb_model.predict(X_test)
    show_metrics(label_encoder.inverse_transform(y_test), label_encoder.inverse_transform(nb_pred))

    print("\n--- Logistic Regression Evaluation ---")
    lr_pred = lr_model.predict(X_test)
    show_metrics(label_encoder.inverse_transform(y_test), label_encoder.inverse_transform(lr_pred))

    # Save the better model (let's assume Logistic Regression for now)
    joblib.dump(lr_model, 'logistic_model.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    joblib.dump(label_encoder, 'label_encoder.joblib')

    while True:
        try:
            user_input = input("\nEnter a review (or type 'exit'): ")
            if user_input.lower() == 'exit':
                break
            clean_input = clean_text(user_input)
            vec = vectorizer.transform([clean_input])
            pred_numeric = lr_model.predict(vec)[0]
            result = label_encoder.inverse_transform([pred_numeric])[0]
            print("Prediction:", result)
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()
