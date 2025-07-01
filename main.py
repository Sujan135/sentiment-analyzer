from data.dataset import load_data
from models.logistic_regression import LogisticRegressionModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from utils.metrics import show_metrics
from collections import Counter
from sklearn.metrics import accuracy_score

def main():
    data = load_data()
    texts = [x[0] for x in data]
    labels = [x[1] for x in data]
    print("Class distribution:", Counter(labels))

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.3, random_state=42, stratify=labels
    )

    model = LogisticRegressionModel()
    model.train(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Actual labels:", y_test)
    print("Predicted labels:", y_pred.tolist())
    show_metrics(y_test, y_pred)

    # Bonus: train & test on full data to check model power
    model.train(X, labels)
    y_pred_full = model.predict(X)
    full_acc = accuracy_score(labels, y_pred_full)
    print("Full data accuracy:", full_acc)
    show_metrics(labels, y_pred_full)

    while True:
        user_input = input("\nEnter a review (or type 'exit'): ")
        if user_input.lower() == 'exit':
            break
        vec = vectorizer.transform([user_input])
        result = model.predict(vec)[0]
        print("Prediction:", result)

if __name__ == "__main__":
    main()
