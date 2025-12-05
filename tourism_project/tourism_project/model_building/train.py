# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score
)
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()


Xtrain_path = "hf://datasets/sgpai/tourism-mlops/X_train.csv"
Xtest_path = "hf://datasets/sgpai/tourism-mlops/X_test.csv"
ytrain_path = "hf://datasets/sgpai/tourism-mlops/y_train.csv"
ytest_path = "hf://datasets/sgpai/tourism-mlops/y_test.csv"

X_train = pd.read_csv(Xtrain_path)
X_test = pd.read_csv(Xtest_path)
y_train = pd.read_csv(ytrain_path)
y_test = pd.read_csv(ytest_path)


numeric_features = ['Age','DurationOfPitch','MonthlyIncome']
categorical_features = ['TypeofContact','Occupation','Gender','ProductPitched','PreferredPropertyStar','MaritalStatus','Designation']

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost Regressor
xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)

# Hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [100, 200],
    'xgbclassifier__learning_rate': [0.01, 0.1],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__min_child_weight': [1, 3],
    'xgbclassifier__gamma': [0, 0.1],
    'xgbclassifier__subsample': [0.7, 0.9]
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    scorer = metrics.make_scorer(metrics.precision_score)
    # Grid Search
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring=scorer)
    grid_search.fit(X_train, y_train)

    # Log parameter sets
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("precision_score", mean_score)

    # Best model
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    train_recal = recall_score(y_train, y_pred_train)
    test_recal = recall_score(y_test, y_pred_test)

    train_prec = precision_score(y_train, y_pred_train)
    test_prec = precision_score(y_test, y_pred_test)

    train_f1 = f1_score(y_train, y_pred_train)
    test_f1 = f1_score(y_test, y_pred_test)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_recall": train_recal,
        "test_recall": test_recal,
        "train_precision": train_prec,
        "test_precision": test_prec,
        "train_f1score":train_f1,
        "test_f1score":test_f1
    })

    # Save the model locally
    model_path = "tourism_package_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "sgpai/tourism-mlops"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="tourism_package_model_v1.joblib",
        path_in_repo="tourism_package_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
