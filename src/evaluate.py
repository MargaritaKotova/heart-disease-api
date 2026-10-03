"""Evaluate on the held-out test set."""
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0,
        ),
    }
