from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import tensorflow as tf
from joblib import dump
from scipy import stats
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor,
                              ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

from bmstu_graduate_project.src.data.make_dataset import (get_data_frame,
                                                          get_X_y)
from bmstu_graduate_project.src.utils.trim_string import trim_string

print(f"scikit-learn Version: {sklearn.__version__}")
print(f"TensorFlow Version: {tf.__version__}")


def plot_model_train_val_losses(history_dict: dict[str, list[float]]) -> None:
    """
    Plots Train & Validation Losses per Epoch
    """
    plt.figure(figsize=(8, 5))
    for array in history_dict.values():
        plt.plot(array)

    plt.title("Loss Plot")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend(history_dict.keys())
    plt.grid()
    plt.show()


def plot_multi_output_solver(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    solver,
    size: int = 50,
    alpha: float = .4
) -> None:
    assert y_test.shape[1] == 2 and y_pred.shape[1] == 2

    caption = solver.get_params()["estimator"] if isinstance(
        solver, MultiOutputRegressor
    ) else type(solver).__name__

    plt.figure()
    plt.scatter(
        y_test[:, 0],
        y_test[:, 1],
        edgecolor="k",
        c="navy",
        s=size,
        marker="s",
        alpha=alpha,
        label="Data",
    )
    plt.scatter(
        y_pred[:, 0],
        y_pred[:, 1],
        edgecolor="k",
        c="cornflowerblue",
        s=size,
        alpha=alpha,
        label=(
            f"MAE: {maes[caption]:,.6f}; "
            f"MSE: {mses[caption]:,.6f}; "
            f"$R^2$: {r2_s[caption]:,.6f}"
        )
    )
    plt.xlim([-3, 3])
    plt.ylim([-2, 4])
    plt.xlabel("depth")
    plt.ylabel("width")
    plt.title(caption)
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()


# =============================================================================
# Constants
# =============================================================================
DESCRIPTION = {
    'IW': 'Величина сварочного тока',
    'IF': 'Ток фокусировки электронного пучка',
    'VW': 'Скорость сварки',
    'FP': 'Расстояние от поверхности образцов до электронно-оптической системы',
    'Depth': 'Глубина шва',
    'Width': 'Ширина шва'
}
CV = 5
RANDOM_STATE = 42
MODEL_DIR = "../models"

# =============================================================================
# Data Collection
# =============================================================================
df = get_data_frame()

# =============================================================================
# Data Preprocessing
# =============================================================================
scaler = StandardScaler()
df_scaled = pd.DataFrame(data=scaler.fit_transform(df), columns=df.columns)

# =============================================================================
# Exploratory Data Analysis, EDA
# =============================================================================
df.describe()
df.info()

# =============================================================================
# Variance Inflation Factor
# =============================================================================
vif = pd.DataFrame(
    data=(
        (column, variance_inflation_factor(df_scaled.values, _))
        for _, column in enumerate(df_scaled.columns)
    ),
    columns=("features", "vif_Factor")
)
# vif

with sns.axes_style('darkgrid'):
    with sns.plotting_context('notebook', font_scale=1.5):
        # =====================================================================
        # Box Plot
        # =====================================================================
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df_scaled, orient="h")

        # =====================================================================
        # Distribution Plot
        # =====================================================================
        for column in df_scaled.columns:
            sns.displot(data=df_scaled, x=column, kde=True)

        # =====================================================================
        # Correlation Matrix
        # =====================================================================
        plt.figure(figsize=(8, 5))
        sns.heatmap(data=df_scaled.corr(), cmap="YlGnBu", annot=True)

        # =====================================================================
        # Pair Plot
        # =====================================================================
        plt.figure(figsize=(8, 5))
        sns.pairplot(data=df_scaled, diag_kind="kde")

        # =====================================================================
        # Violin Plot
        # =====================================================================
        plt.figure(figsize=(8, 5))
        sns.violinplot(data=df_scaled)


# $\sigma = 1.4$ is the Optimum for Trimming In Terms of Outliers
df_trimmed = df_scaled[(np.abs(stats.zscore(df_scaled)) < 1.4).all(axis=1)]
# df_trimmed


# =============================================================================
# ML Models -- scikit-learn
# =============================================================================
# =============================================================================
# Get Features
# =============================================================================
X, y = df_scaled.pipe(get_X_y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8)

# =============================================================================
# scikit-learn: Linear Models
# =============================================================================

maes, mses, r2_s = {}, {}, {}

for solver in (
    ElasticNetCV(cv=CV, random_state=RANDOM_STATE),
    LassoCV(cv=CV, random_state=RANDOM_STATE),
    RidgeCV(cv=CV)
):
    regr_multirf = MultiOutputRegressor(solver)
    regr_multirf.fit(X_train, y_train)
    y_pred = regr_multirf.predict(X_test)

    maes[type(solver).__name__] = mean_absolute_error(y_test, y_pred)
    mses[type(solver).__name__] = mean_squared_error(y_test, y_pred)
    r2_s[type(solver).__name__] = r2_score(y_test, y_pred)

    # =========================================================================
    # Plot the Results
    # =========================================================================
    plot_multi_output_solver(y_test, y_pred, solver)

    # =========================================================================
    # Save Model
    # =========================================================================
    file_name = f'sklearn_linear_models_{type(solver).__name__.lower()}.joblib'
    dump(solver, Path(MODEL_DIR).joinpath(file_name))


best_mae = min(maes, key=maes.get)
best_mse = min(mses, key=mses.get)
best_r_2 = max(r2_s, key=r2_s.get)
print("Linear Models Results:")
print(f"Best Solver In Terms of <MAE> Is: {best_mae} = {maes[best_mae]:,.6f}")
print(f"Best Solver In Terms of <MSE> Is: {best_mse} = {mses[best_mse]:,.6f}")
print(f"Best Solver In Terms of <R**2> Is: {best_r_2} = {r2_s[best_r_2]:,.6f}")


# =============================================================================
# scikit-learn: Ensembles
# =============================================================================
maes, mses, r2_s = {}, {}, {}

for solver in (
    AdaBoostRegressor(random_state=RANDOM_STATE),
    BaggingRegressor(random_state=RANDOM_STATE),
    ExtraTreesRegressor(random_state=RANDOM_STATE),
    GradientBoostingRegressor(random_state=RANDOM_STATE),
    HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    RandomForestRegressor(random_state=RANDOM_STATE),
):
    regressor = MultiOutputRegressor(solver)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    maes[type(solver).__name__] = mean_absolute_error(y_test, y_pred)
    mses[type(solver).__name__] = mean_squared_error(y_test, y_pred)
    r2_s[type(solver).__name__] = r2_score(y_test, y_pred)

    # =========================================================================
    # Plot the Results
    # =========================================================================
    plot_multi_output_solver(y_test, y_pred, solver)

    # =========================================================================
    # Save Model
    # =========================================================================
    file_name = f'sklearn_ensembles_{type(solver).__name__.lower()}.joblib'
    dump(solver, Path(MODEL_DIR).joinpath(file_name))


best_mae = min(maes, key=maes.get)
best_mse = min(mses, key=mses.get)
best_r_2 = max(r2_s, key=r2_s.get)
print("Ensembles Results:")
print(f"Best Solver In Terms of <MAE> Is: {best_mae} = {maes[best_mae]:,.6f}")
print(f"Best Solver In Terms of <MSE> Is: {best_mse} = {mses[best_mse]:,.6f}")
print(f"Best Solver In Terms of <R**2> Is: {best_r_2} = {r2_s[best_r_2]:,.6f}")


# =============================================================================
# scikit-learn: Grid Search
# =============================================================================
ESTIMATORS = {
    BaggingRegressor(random_state=RANDOM_STATE): {
        "estimator__n_estimators": [10, 20, 25, 50, 100],
        # "estimator__max_samples": [12, 14, 18, 24],
        # "estimator__max_features": [2, 3, 4],
    },
    GradientBoostingRegressor(random_state=RANDOM_STATE): {
        # "estimator__loss": ['squared_error', 'absolute_error', 'huber', 'quantile'],
        "estimator__n_estimators": [10, 20, 25, 50, 100],
        # "estimator__min_samples_split": [2, 3, 4, 5],
        # "estimator__min_samples_leaf": [1, 2, 3, 4],
        # "estimator__max_depth": [1, 2, 3, 4],
        # "estimator__max_features": ['sqrt', 'log2'],
    },
    RandomForestRegressor(random_state=RANDOM_STATE): {
        "estimator__n_estimators": [10, 20, 25, 50, 100],
        # "estimator__criterion": ['squared_error', 'absolute_error', 'poisson'],
        # "estimator__min_samples_split": [2, 3, 4, 5],
        # "estimator__min_samples_leaf": [1, 2, 3, 4],
        # "estimator__max_features": [2, 3, 4],
    },
}

maes, mses, r2_s = {}, {}, {}

for estimator, param_grid in ESTIMATORS.items():
    regressor = MultiOutputRegressor(estimator)
    gscv = GridSearchCV(
        estimator=regressor,
        param_grid=param_grid,
        cv=CV,
        verbose=2,
        n_jobs=-1
    )
    gscv.fit(X_train, y_train)

    print(gscv.best_params_)

    best_estimator = gscv.best_estimator_
    y_pred = best_estimator.predict(X_test)
    print(best_estimator.get_params())

    maes[best_estimator.get_params()["estimator"]] = mean_absolute_error(
        y_test, y_pred
    )
    mses[best_estimator.get_params()["estimator"]] = mean_squared_error(
        y_test, y_pred
    )
    r2_s[best_estimator.get_params()["estimator"]] = r2_score(
        y_test, y_pred
    )
    # =========================================================================
    # Plot the Results
    # =========================================================================
    plot_multi_output_solver(y_test, y_pred, best_estimator)

    # =========================================================================
    # Save Model
    # =========================================================================
    file_name = f'sklearn_grid_search_{trim_string(str(best_estimator.get_params()["estimator"]))}.joblib'
    dump(best_estimator, Path(MODEL_DIR).joinpath(file_name))


best_mae = min(maes, key=maes.get)
best_mse = min(mses, key=mses.get)
best_r_2 = max(r2_s, key=r2_s.get)
print("Grid Search Results:")
print(
    f"Best Solver In Terms of <MAE> Is:\n"
    f" {best_mae}\n"
    f" MAE = {maes[best_mae]:,.6f}\n"
)
print(
    f"Best Solver In Terms of <MSE> Is:\n"
    f" {best_mse}\n"
    f" MSE = {mses[best_mse]:,.6f}\n"
)
print(
    f"Best Solver In Terms of <R**2> Is:\n"
    f" {best_r_2}\n"
    f" R**2 = {r2_s[best_r_2]:,.6f}\n"
)


# =============================================================================
# TensorFlow: Deep Learning
# =============================================================================
model = Sequential()
model.add(
    Dense(32,
          input_dim=X.shape[1],
          kernel_initializer="he_uniform",
          activation="relu")
)
model.add(Dropout(.05))
model.add(Dense(y.shape[1]))
model.compile(optimizer="adam", loss="mae")
# =============================================================================
# Architecture
# =============================================================================
model.summary()

model_history = model.fit(X_train, y_train, verbose=1,
                          epochs=100, validation_split=.2)

model.evaluate(X_test, y_test, verbose=0)

# =============================================================================
# Plot the Results
# =============================================================================
plot_model_train_val_losses(model_history.history)

# =============================================================================
# Save Model
# =============================================================================
model.save(MODEL_DIR)
# =============================================================================
# Load Model
# =============================================================================
new_model = tf.keras.models.load_model(MODEL_DIR)
# =============================================================================
# Architecture
# =============================================================================
new_model.summary()
