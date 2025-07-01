import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from utils.metrics import show_metrics
from data.dataset import load_data
import nltk
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.decomposition import TruncatedSVD

nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    corrected = str(TextBlob(text).correct())
    corrected = re.sub(r'\W+', ' ', corrected).strip()
    corrected = handle_negation(corrected)
    words = corrected.split()
    lemmatized = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(lemmatized)

def handle_negation(text):
    negation_words = {'not', "don't", "didn't", "no", "never"}
    tokens = text.split()
    result = []
    negate = False
    for word in tokens:
        if word in negation_words:
            negate = True
            result.append(word)
        elif negate:
            result.append("NOT_" + word)
            negate = False
        else:
            result.append(word)
    return " ".join(result)

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

def weighted_ensemble_vote(preds, weights):
    vote_scores = {}
    for pred, w in zip(preds, weights):
        vote_scores[pred] = vote_scores.get(pred, 0) + w
    return max(vote_scores, key=vote_scores.get)

def main():
    data = load_data()
    texts_raw = [x[0] for x in data]
    raw_labels = [x[1] for x in data]
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(raw_labels)

    print("Class distribution:", Counter(raw_labels))

    texts = [clean_text(t) for t in texts_raw]

    vectorizer_nb = CountVectorizer(ngram_range=(1, 2), stop_words='english')
    X_nb = vectorizer_nb.fit_transform(texts)
    X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(
        X_nb, labels, test_size=0.3, random_state=42, stratify=labels
    )
    nb_model = tune_nb(X_train_nb, y_train_nb)

    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english')
    X_counts = vectorizer.fit_transform(texts)
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_embedded = svd.fit_transform(X_counts)
    X_train_emb, X_test_emb, y_train_emb, y_test_emb = train_test_split(
        X_embedded, labels, test_size=0.3, random_state=42, stratify=labels
    )
    lr_model = tune_logistic_regression(X_train_emb, y_train_emb)
    svm_model = tune_svm(X_train_emb, y_train_emb)

    print("\n--- Naive Bayes Evaluation ---")
    y_pred_nb = nb_model.predict(X_test_nb)
    show_metrics(label_encoder.inverse_transform(y_test_nb), label_encoder.inverse_transform(y_pred_nb))
    acc_nb = accuracy_score(y_test_nb, y_pred_nb)

    print("\n--- Logistic Regression Evaluation ---")
    y_pred_lr = lr_model.predict(X_test_emb)
    show_metrics(label_encoder.inverse_transform(y_test_emb), label_encoder.inverse_transform(y_pred_lr))
    acc_lr = accuracy_score(y_test_emb, y_pred_lr)

    print("\n--- SVM Evaluation ---")
    y_pred_svm = svm_model.predict(X_test_emb)
    show_metrics(label_encoder.inverse_transform(y_test_emb), label_encoder.inverse_transform(y_pred_svm))
    acc_svm = accuracy_score(y_test_emb, y_pred_svm)

    weights = [acc_nb, acc_lr, acc_svm]

    while True:
        user_input = input("\nEnter a review (or type 'exit'): ")
        if user_input.lower() == 'exit':
            break
        clean_input = clean_text(user_input)

        vec_nb = vectorizer_nb.transform([clean_input])
        pred_nb = label_encoder.inverse_transform(nb_model.predict(vec_nb))[0]

        vec_counts = vectorizer.transform([clean_input])
        vec_emb = svd.transform(vec_counts)
        pred_lr = label_encoder.inverse_transform(lr_model.predict(vec_emb))[0]
        pred_svm = label_encoder.inverse_transform(svm_model.predict(vec_emb))[0]

        final_pred = weighted_ensemble_vote([pred_nb, pred_lr, pred_svm], weights)

        print(f"Naive Bayes prediction: {pred_nb}")
        print(f"Logistic Regression prediction: {pred_lr}")
        print(f"SVM prediction: {pred_svm}")
        print(f"Ensemble final prediction: {final_pred}")

if __name__ == "__main__":
    main()
