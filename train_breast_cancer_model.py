import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def main() -> None:
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    model = ExtraTreesClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    print("Validation Accuracy:", round(accuracy_score(y_val, val_pred), 4))
    print("Test Accuracy:", round(accuracy_score(y_test, test_pred), 4))
    print("Test Classification Report:")
    print(classification_report(y_test, test_pred, target_names=dataset.target_names))

    artifact = {
        "model": model,
        "feature_names": list(X.columns),
        "target_names": list(dataset.target_names),  # ['malignant', 'benign']
    }
    joblib.dump(artifact, "breast_cancer_model.pkl")
    print("Saved model artifact: breast_cancer_model.pkl")


if __name__ == "__main__":
    main()
