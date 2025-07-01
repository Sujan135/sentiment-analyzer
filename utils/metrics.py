# metrics.py
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def show_metrics(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")

    cm = confusion_matrix(y_test, y_pred, labels=["positive", "negative"])
    sns.heatmap(cm, annot=True, xticklabels=["positive", "negative"], yticklabels=["positive", "negative"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
