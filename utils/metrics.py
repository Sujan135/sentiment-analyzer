from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def show_metrics(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")

    # Use numeric labels here, matching LabelEncoder output
    labels = [0, 1]
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Map numeric labels to strings for plot ticks
    label_names = ["negative", "positive"]  # assuming 0=negative, 1=positive

    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
