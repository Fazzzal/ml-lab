import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.tools.feature_tools import apply_feature_engineering


# Single source of truth for which model names are actually
# executable. Factories (not instances) so every experiment gets
# a fresh, unfitted estimator.
CLASSIFICATION_MODEL_FACTORIES = {
    "Logistic Regression":
        lambda: LogisticRegression(max_iter=1000),
    "Random Forest":
        lambda: RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
    "Gradient Boosting":
        lambda: GradientBoostingClassifier(
            random_state=42
        ),
}

REGRESSION_MODEL_FACTORIES = {
    "Linear Regression":
        lambda: LinearRegression(),
    "Random Forest Regressor":
        lambda: RandomForestRegressor(
            n_estimators=100,
            random_state=42,
        ),
    "Gradient Boosting Regressor":
        lambda: GradientBoostingRegressor(
            random_state=42
        ),
}

# Hyperparameters the critic is allowed to request per model. This
# is an allow-list, not free-form kwargs, so a hallucinated or
# malicious parameter name is simply dropped rather than passed
# through to sklearn.
ALLOWED_PARAMETERS = {
    "Logistic Regression": {"C"},
    "Random Forest": {"n_estimators", "max_depth", "min_samples_leaf"},
    "Gradient Boosting": {"n_estimators", "max_depth", "learning_rate"},
    "Linear Regression": set(),
    "Random Forest Regressor": {"n_estimators", "max_depth", "min_samples_leaf"},
    "Gradient Boosting Regressor": {"n_estimators", "max_depth", "learning_rate"},
}


def get_supported_models(problem_type: str) -> list[str]:
    """Return the list of model names that can actually be
    executed for a given problem type. Every agent that receives
    a model name from an LLM should validate against this before
    trusting it."""

    if problem_type == "classification":
        return list(CLASSIFICATION_MODEL_FACTORIES.keys())

    if problem_type == "regression":
        return list(REGRESSION_MODEL_FACTORIES.keys())

    return []


def build_preprocessor(X):
    numeric_columns = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    transformers = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    return ColumnTransformer(
        transformers
    )


def get_model(
    model_name,
    problem_type
):
    if problem_type == "classification":
        factories = CLASSIFICATION_MODEL_FACTORIES
    else:
        factories = REGRESSION_MODEL_FACTORIES

    if model_name not in factories:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    return factories[model_name]()


def apply_parameters(model, model_name, parameters):
    """Set only the allow-listed hyperparameters on `model`.
    Returns (applied, rejected) dicts/sets for logging/reporting."""

    parameters = parameters or {}
    allowed = ALLOWED_PARAMETERS.get(model_name, set())

    applied = {
        key: value
        for key, value in parameters.items()
        if key in allowed
    }

    rejected = sorted(set(parameters.keys()) - allowed)

    if applied:
        model.set_params(**applied)

    return applied, rejected


def _load_and_clean(dataset_path, dataframe=None):
    if dataframe is not None:
        df = dataframe.copy()
    else:
        df = pd.read_csv(dataset_path)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )

    return df


def _prepare_xy(df, target_column, problem_type):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    if problem_type == "classification" and y.dtype == "object":
        y = y.astype(str).str.strip()

        if set(y.unique()) == {"Yes", "No"}:
            y = y.map({"No": 0, "Yes": 1})

    return X, y


def run_experiment(
    dataset_path,
    target_column,
    model_name,
    problem_type,
    train_indices,
    val_indices,
    parameters=None,
    cv_folds=0,
    dataframe=None,
):
    """Train on train_indices, report metrics on val_indices. This
    is the metric the critic loop compares experiments against —
    it never touches test_indices, which is reserved for
    final_report_agent's one-time holdout evaluation."""

    df = _load_and_clean(dataset_path, dataframe)
    X, y = _prepare_xy(df, target_column, problem_type)

    preprocessor = build_preprocessor(X)
    model = get_model(model_name, problem_type)

    applied_params, rejected_params = apply_parameters(
        model, model_name, parameters
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_val, y_val = X.iloc[val_indices], y.iloc[val_indices]

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_val)

    cv_result = None

    if cv_folds and cv_folds > 1:
        scoring = (
            "f1_weighted"
            if problem_type == "classification"
            else "neg_root_mean_squared_error"
        )

        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv_folds, scoring=scoring,
        )

        cv_result = {
            "folds": cv_folds,
            "scoring": scoring,
            "mean": float(cv_scores.mean()),
            "std": float(cv_scores.std()),
        }

    base_result = {
        "model": model_name,
        "problem_type": problem_type,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "applied_params": applied_params,
        "rejected_params": rejected_params,
        "cross_validation": cv_result,
    }

    if problem_type == "classification":
        base_result["accuracy"] = float(
            accuracy_score(y_val, predictions)
        )
        base_result["f1"] = float(
            f1_score(y_val, predictions, average="weighted")
        )

        return base_result

    mse = mean_squared_error(y_val, predictions)

    base_result["rmse"] = float(mse ** 0.5)
    base_result["r2"] = float(r2_score(y_val, predictions))

    return base_result


def run_final_holdout_evaluation(
    dataset_path,
    target_column,
    model_name,
    problem_type,
    train_indices,
    val_indices,
    test_indices,
    parameters=None,
    feature_type=None,
    feature_name=None,
):
    """Retrain the chosen model on train+val combined, then
    evaluate exactly once on test_indices. Called only by
    final_report_agent, after the critic loop has already decided
    it is done."""

    if feature_type and feature_type != "none":
        df = apply_feature_engineering(
            dataset_path=dataset_path,
            target_column=target_column,
            feature_type=feature_type,
            feature_name=feature_name or feature_type,
        )
    else:
        df = _load_and_clean(dataset_path)

    X, y = _prepare_xy(df, target_column, problem_type)

    preprocessor = build_preprocessor(X)
    model = get_model(model_name, problem_type)
    applied_params, rejected_params = apply_parameters(
        model, model_name, parameters
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    train_final_indices = list(train_indices) + list(val_indices)

    X_train, y_train = (
        X.iloc[train_final_indices], y.iloc[train_final_indices]
    )
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    result = {
        "model": model_name,
        "problem_type": problem_type,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "applied_params": applied_params,
        "rejected_params": rejected_params,
    }

    if problem_type == "classification":
        result["accuracy"] = float(
            accuracy_score(y_test, predictions)
        )
        result["f1"] = float(
            f1_score(y_test, predictions, average="weighted")
        )

        return result

    mse = mean_squared_error(y_test, predictions)

    result["rmse"] = float(mse ** 0.5)
    result["r2"] = float(r2_score(y_test, predictions))

    return result
