from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def show_metrics(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")

    # Find unique labels in y_test and y_pred combined, sorted for consistency
    unique_labels = sorted(list(set(y_test) | set(y_pred)))

    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

    sns.heatmap(cm, annot=True, fmt='d', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
