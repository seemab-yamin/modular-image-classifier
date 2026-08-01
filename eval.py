"""
Evaluation Utilities - Metrics, Confusion Matrix
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


# ============================================================
# BLOCK 1: METRICS COMPUTATION
# ============================================================
def evaluate(all_labels, all_preds, class_names):
    """
    Compute all classification metrics.

    Returns:
        dict: Contains macro_f1, weighted_acc, report, cm,
              precision, recall, f1, support
    """
    # 1. Classification report
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
    )

    # 2. Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # 3. Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average=None,
    )

    return {
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_acc": report["weighted avg"]["precision"],
        "report": report,
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


# ============================================================
# BLOCK 2: CONFUSION MATRIX
# ============================================================
def save_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Confusion matrix saved: {save_path}")


# ============================================================
# BLOCK 3: SUMMARY REPORT
# ============================================================
def save_summary_report(report, class_names, model_arch, dataset, save_path):
    """
    Save summary report with class-wise and global metrics.

    Appends each experiment block to the file.
    """
    # Class-wise data
    class_data = []
    for name in class_names:
        class_data.append(
            {
                "Class": name,
                "Precision": report[name]["precision"],
                "Recall": report[name]["recall"],
                "F1-Score": report[name]["f1-score"],
                "Support": int(report[name]["support"]),
            }
        )
    class_df = pd.DataFrame(class_data)

    # Global aggregates
    global_data = {
        "Metric": ["Macro F1", "Weighted Precision", "Weighted Recall", "Weighted F1"],
        "Score": [
            report["macro avg"]["f1-score"],
            report["weighted avg"]["precision"],
            report["weighted avg"]["recall"],
            report["weighted avg"]["f1-score"],
        ],
    }
    global_df = pd.DataFrame(global_data)

    # Write to file (append mode)
    with open(save_path, "a") as f:
        f.write(f"Experiment: {model_arch} | Dataset: {dataset}\n")
        f.write("-" * 60 + "\n")
        f.write("\nClass-wise Metrics:\n")
        f.write(class_df.to_string(index=False))
        f.write("\n\n")
        f.write("Global Metrics:\n")
        f.write(global_df.to_string(index=False))
        f.write("\n")
        f.write("=" * 60 + "\n\n")

    print(f"✅ Summary report appended: {save_path}")


# ============================================================
# BLOCK 4: PER-CLASS METRICS TABLE
# ============================================================
def save_per_class_metrics(precision, recall, f1, support, class_names, save_path):
    """Save per-class metrics as CSV."""
    data = {
        "Class": class_names,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Support": support,
    }
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"✅ Per-class metrics saved: {save_path}")
    return df


# ============================================================
# BLOCK 5: PRINT METRICS
# ============================================================
def print_metrics(metrics, class_names):
    """Pretty print metrics."""
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted Accuracy: {metrics['weighted_acc']:.4f}")
    print("\nPer-class metrics:")
    for i, name in enumerate(class_names):
        print(
            f"  {name:15s} | P: {metrics['precision'][i]:.3f} "
            f"R: {metrics['recall'][i]:.3f} F1: {metrics['f1'][i]:.3f}"
        )
    print("=" * 60 + "\n")


# ============================================================
# BLOCK 6: RUN STANDALONE (Optional)
# ============================================================
if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Dummy data
    all_labels = np.random.randint(0, 10, 100)
    all_preds = np.random.randint(0, 10, 100)
    class_names = [f"Class_{i}" for i in range(10)]

    metrics = evaluate(all_labels, all_preds, class_names)
    print_metrics(metrics, class_names)
