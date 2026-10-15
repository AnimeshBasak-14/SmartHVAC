"""
models.py - the list of ML models we compare.

Models that are sensitive to the scale of numbers (Linear, KNN, SVR, MLP,
Logistic Regression) are wrapped in a Pipeline with StandardScaler, so the
scaling is learned from the training data only (no data leakage).
"""
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.config import RANDOM_STATE

RS = RANDOM_STATE


def regression_models(quick=False):
    """Models that predict a NUMBER (heating load or cooling load)."""
    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "K-Nearest Neighbours": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
        "Decision Tree": DecisionTreeRegressor(random_state=RS),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RS, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RS),
        "Support Vector Regression": make_pipeline(StandardScaler(), SVR(C=100, epsilon=0.1)),
        "Neural Network (MLP)": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=3000, early_stopping=True, random_state=RS)),
    }
    if quick:   # fast mode for testing the pipeline
        models = {k: models[k] for k in ["Linear Regression", "Decision Tree", "Gradient Boosting"]}
    return models


def classification_models(quick=False):
    """Models that predict a CLASS (Low / Medium / High energy demand)."""
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "K-Nearest Neighbours": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Decision Tree": DecisionTreeClassifier(random_state=RS),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RS, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RS),
    }
    if quick:
        models = {k: models[k] for k in ["Logistic Regression", "Decision Tree"]}
    return models


# Small hyper-parameter grids, used to fine-tune the best regression model.
# Keys must match the model names above.
PARAM_GRIDS = {
    "Random Forest": {
        "n_estimators": [200, 500],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2],
    },
    "Gradient Boosting": {
        "n_estimators": [200, 500],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3, 4],
    },
    "Decision Tree": {
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5],
    },
    "K-Nearest Neighbours": {
        "kneighborsregressor__n_neighbors": [3, 5, 7, 10],
        "kneighborsregressor__weights": ["uniform", "distance"],
    },
    "Support Vector Regression": {
        "svr__C": [10, 100, 1000],
        "svr__epsilon": [0.05, 0.1, 0.5],
    },
}
