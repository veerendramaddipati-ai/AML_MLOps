
import pandas as pd
import joblib
import mlflow

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ---------------------------
# Load Train-Test Data
# ---------------------------

X_train = pd.read_csv("Xtrain.csv")
X_test = pd.read_csv("Xtest.csv")

y_train = pd.read_csv("ytrain.csv").squeeze()
y_test = pd.read_csv("ytest.csv").squeeze()

# ---------------------------
# Identify Column Types
# ---------------------------

cat_cols = X_train.select_dtypes(include="object").columns
num_cols = X_train.select_dtypes(exclude="object").columns

# ---------------------------
# Preprocessing
# ---------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# ---------------------------
# Model
# ---------------------------

rf = RandomForestClassifier(random_state=42)

pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("model", rf)
    ]
)

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 10, None]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1
)

mlflow.set_experiment("Tourism_MLOps_Project")

with mlflow.start_run():

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    preds = best_model.predict(X_test)
    prob = best_model.predict_proba(X_test)[:,1]

    accuracy = accuracy_score(y_test,preds)
    precision = precision_score(y_test,preds)
    recall = recall_score(y_test,preds)
    f1 = f1_score(y_test,preds)
    roc_auc = roc_auc_score(y_test,prob)

    mlflow.log_params(grid.best_params_)

    mlflow.log_metric("accuracy",accuracy)
    mlflow.log_metric("precision",precision)
    mlflow.log_metric("recall",recall)
    mlflow.log_metric("f1_score",f1)
    mlflow.log_metric("roc_auc",roc_auc)

    # Save the entire pipeline, not just the model and preprocessor separately
    joblib.dump(
        best_model,
        "tourism_project/deployment/best_model.pkl"
    )

    print("Model saved successfully")
    print("Best Parameters")
    print(grid.best_params_)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"ROC_AUC  : {roc_auc:.4f}")

    print("Model Saved Successfully")
