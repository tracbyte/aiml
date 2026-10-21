import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

if __name__ == "__main__":
    tp, fn, fp, tn = 120, 30, 25, 150

    y_true = np.array([1] * (tp + fn) + [0] * (fp + tn))
    y_pred = np.array([1] * tp + [0] * fn + [1] * fp + [0] * tn)

    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")

    cm = np.array([[tp, fn], [fp, tn]])

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Positive (Predicted)", "Negative (Predicted)"],
        yticklabels=["Positive (Actual)", "Negative (Actual)"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()
