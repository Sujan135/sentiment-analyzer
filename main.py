# main.py
from data.dataset import load_data
from models.naive_bayes import NaiveBayesModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from utils.metrics import show_metrics

def main():
    # Load dataset
    data = load_data()
    texts = [x[0] for x in data]
    labels = [x[1] for x in data]

    # Text -> numbers
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

    # Train model
    model = NaiveBayesModel()
    model.train(X_train, y_train)

    # Test model
    y_pred = model.predict(X_test)
    show_metrics(y_test, y_pred)

    # Predict custom input
    while True:
        user_input = input("\nEnter a review (or type 'exit'): ")
        if user_input.lower() == 'exit':
            break
        vec = vectorizer.transform([user_input])
        result = model.predict(vec)[0]
        print("Prediction:", result)

if __name__ == "__main__":
    main()