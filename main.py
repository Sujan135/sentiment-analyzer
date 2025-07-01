import re
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from utils.metrics import show_metrics
from nltk.corpus import stopwords

from data.dataset import load_data

nltk.download('stopwords')

NEGATION_WORDS = {
    "not", "never", "no", "none", "nobody", "nothing", "neither", 
    "nowhere", "hardly", "scarcely", "barely"
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text).strip()
    words = text.split()

    result = []
    negation_scope = 0

    for word in words:
        if word in NEGATION_WORDS:
            negation_scope = 3  # next 3 words negated
            result.append(word)
        elif negation_scope > 0:
            result.append(word + "_NEG")
            negation_scope -= 1
        else:
            result.append(word)

    return ' '.join(result)

def tune_nb(X_train, y_train):
    nb = MultinomialNB()
    params = {'alpha': [0.1, 0.5]}
    grid = GridSearchCV(nb, params, cv=3, scoring='accuracy')
    grid.fit(X_train, y_train)
    print("Best Naive Bayes alpha:", grid.best_params_['alpha'])
    return grid.best_estimator_

def tune_logistic_regression(X_train, y_train):
    lr = LogisticRegression(solver='liblinear', max_iter=500)
    params = {'C': [1, 10]}
    grid = GridSearchCV(lr, params, cv=3, scoring='accuracy')
    grid.fit(X_train, y_train)
    print("Best Logistic Regression C:", grid.best_params_['C'])
    return grid.best_estimator_

def tune_svm(X_train, y_train):
    svm = SVC(kernel='linear')
    params = {'C': [1, 10]}
    grid = GridSearchCV(svm, params, cv=3, scoring='accuracy')
    grid.fit(X_train, y_train)
    print("Best SVM C:", grid.best_params_['C'])
    return grid.best_estimator_

def main():
    data = load_data()
    texts = [clean_text(x[0]) for x in data]
    raw_labels = [x[1] for x in data]
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(raw_labels)

    print("Class distribution:", Counter(raw_labels))

    stop_words = stopwords.words('english')
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words=stop_words,
        sublinear_tf=True,
        min_df=2,
        max_df=0.95
    )
    X = vectorizer.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.3, random_state=42, stratify=labels
    )

    nb_model = tune_nb(X_train, y_train)
    lr_model = tune_logistic_regression(X_train, y_train)
    svm_model = tune_svm(X_train, y_train)

    print("\n--- Naive Bayes Evaluation ---")
    y_pred_nb = nb_model.predict(X_test)
    show_metrics(label_encoder.inverse_transform(y_test), label_encoder.inverse_transform(y_pred_nb))

    print("\n--- Logistic Regression Evaluation ---")
    y_pred_lr = lr_model.predict(X_test)
    show_metrics(label_encoder.inverse_transform(y_test), label_encoder.inverse_transform(y_pred_lr))

    print("\n--- SVM Evaluation ---")
    y_pred_svm = svm_model.predict(X_test)
    show_metrics(label_encoder.inverse_transform(y_test), label_encoder.inverse_transform(y_pred_svm))

    while True:
        user_input = input("\nEnter a review (or type 'exit'): ")
        if user_input.lower() == 'exit':
            break
        clean_input = clean_text(user_input)
        vec = vectorizer.transform([clean_input])

        pred_nb = label_encoder.inverse_transform(nb_model.predict(vec))[0]
        pred_lr = label_encoder.inverse_transform(lr_model.predict(vec))[0]
        pred_svm = label_encoder.inverse_transform(svm_model.predict(vec))[0]

        print(f"Naive Bayes prediction: {pred_nb}")
        print(f"Logistic Regression prediction: {pred_lr}")
        print(f"SVM prediction: {pred_svm}")

if __name__ == "__main__":
    main()
