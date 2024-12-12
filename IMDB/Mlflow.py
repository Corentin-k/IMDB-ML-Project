import pandas as pd
import matplotlib.pyplot as plt
import spacy
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
import seaborn as sns
from sklearn.pipeline import Pipeline
from scipy.stats import loguniform
print("MLflow :")

nlp = spacy.load('en_core_web_sm')

dfClean = pd.read_csv('dfClean2.csv')



vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=50000, stop_words='english')
X = vectorizer.fit_transform(dfClean['review'])
y = dfClean['sentiment']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


param_dist = {
    'C': loguniform(1e-2, 10),
    'solver': ['liblinear', 'lbfgs'],
}

model = LogisticRegression(random_state=42)


random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    cv=5,
    scoring='accuracy',
    random_state=42
)


if mlflow.active_run():
    mlflow.end_run()


mlflow.set_experiment("Sentiment Analysis IMDB 6")


random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

metrics_history = []
input_example = pd.DataFrame({'review': ["This movie was fantastic! A must-watch."]})
input_example = vectorizer.transform(input_example['review'])

print("Best Parameters: ", random_search.best_params_)
if mlflow.active_run():
    mlflow.end_run()

#Save each model
for i, params in enumerate(random_search.cv_results_['params']):
    with mlflow.start_run(nested=True):
        model = LogisticRegression(**params, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]


        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label="positive")
        recall = recall_score(y_test, y_pred, pos_label="positive")
        f1 = f1_score(y_test, y_pred, pos_label="positive")
        auc = roc_auc_score(y_test, y_proba)


        metrics_history.append((i, accuracy, precision, recall, f1, auc))


        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc", auc)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f"logistic_regression_model_{i}",
            input_example=input_example,
            signature=False
        )

metrics_df = pd.DataFrame(metrics_history, columns=['Index', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])
metrics_df.set_index('Index', inplace=True)

plt.figure(figsize=(10, 6))
metrics_df[['Accuracy', 'Precision', 'Recall', 'F1']].plot(marker='o')
plt.title("Comparaison des Performances des Modèles")
plt.xlabel("Index du Modèle")
plt.ylabel("Score")
plt.grid(True)
plt.legend(loc="lower right")
plt.savefig("model_performance_comparison.png")
mlflow.log_artifact("model_performance_comparison.png")
plt.close()

metrics_df.to_html("model_performance_report.html")
mlflow.log_artifact("model_performance_report.html")

if mlflow.active_run():
    mlflow.end_run()

with mlflow.start_run():
    best_model = random_search.best_estimator_

    full_pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('model', best_model)
    ])

    mlflow.log_params(random_search.best_params_)
    mlflow.sklearn.log_model(
        sk_model=full_pipeline,
        artifact_path="best_full_pipeline",
        input_example=input_example,
        signature=False
    )

    y_pred_best = best_model.predict(X_test)
    y_proba_best = best_model.predict_proba(X_test)[:, 1]

    best_accuracy = accuracy_score(y_test, y_pred_best)
    best_precision = precision_score(y_test, y_pred_best, pos_label="positive")
    best_recall = recall_score(y_test, y_pred_best, pos_label="positive")
    best_f1 = f1_score(y_test, y_pred_best, pos_label="positive")
    best_auc = roc_auc_score(y_test, y_proba_best)


    mlflow.log_metric("best_accuracy", best_accuracy)
    mlflow.log_metric("best_precision", best_precision)
    mlflow.log_metric("best_recall", best_recall)
    mlflow.log_metric("best_f1_score", best_f1)
    mlflow.log_metric("best_auc", best_auc)

    predictions_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred_best
    })
    predictions_df.to_csv("predictions.csv", index=False)
    mlflow.log_artifact("predictions.csv")