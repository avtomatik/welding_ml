import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from scipy import stats
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor,
                              ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor,
                              RandomForestRegressor, StackingRegressor,
                              VotingRegressor)
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .config import RANDOM_STATE
from .dataset import get_data_frame
from .features import get_X_y

print(f'scikit-learn Version: {sklearn.__version__}')


def get_X(df: pd.DataFrame) -> np.ndarray:
    n_columns = 4
    return df.iloc[:, :n_columns].values
    # return df.iloc[:, :n_columns].values[:, np.newaxis]


def get_y(df: pd.DataFrame, regressor: str = 'depth') -> np.ndarray:
    return df.loc[:, regressor.title()].values


def fit_linear_cv_model(X, regressor: str, model):
    y = df_scaled.pipe(get_y, regressor=regressor)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8)

    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    mse = mean_squared_error(y_test, model.predict(X_test))
    r2_ = r2_score(y_test, model.predict(X_test))
    print(model.alpha_)
    print(model.intercept_)
    print(f'MAE: {mae:,.6f}')
    print(f'MSE: {mse:,.6f}')
    print(f'R^2: {r2_:,.6f}')


# =============================================================================
# Data Collection
# =============================================================================
df = get_data_frame()

# =============================================================================
# Exploratory Data Analysis, EDA
# =============================================================================
df.info()
print(df.describe())

# =============================================================================
# Data Preprocessing: Create Scaled DataFrame
# =============================================================================
scaler = StandardScaler()
df_scaled = pd.DataFrame(data=scaler.fit_transform(df), columns=df.columns)

# =============================================================================
# Variance Inflation Factor
# =============================================================================
vif = pd.DataFrame(
    data=(
        (column, variance_inflation_factor(df_scaled.values, _))
        for _, column in enumerate(df_scaled.columns)
    ),
    columns=('features', 'vif_Factor')
)
# vif

with sns.axes_style('darkgrid'):
    with sns.plotting_context('notebook', font_scale=1.5):
        # =====================================================================
        # Box Plot
        # =====================================================================
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df_scaled, orient='h')

        # =====================================================================
        # Distribution Plot
        # =====================================================================
        for column in df_scaled.columns:
            sns.displot(data=df_scaled, x=column, kde=True)

        # =====================================================================
        # Correlation Matrix
        # =====================================================================
        plt.figure(figsize=(8, 5))
        sns.heatmap(data=df_scaled.corr(), cmap='YlGnBu', annot=True)

        # =====================================================================
        # Pair Plot
        # =====================================================================
        plt.figure(figsize=(8, 5))
        sns.pairplot(data=df_scaled, diag_kind='kde')

        # =====================================================================
        # Violin Plot
        # =====================================================================
        plt.figure(figsize=(8, 5))
        sns.violinplot(data=df_scaled)


# $\sigma = 1.4$ is the Optimum for Trimming In Terms of Outliers
df_trimmed = df_scaled[(np.abs(stats.zscore(df_scaled)) < 1.4).all(axis=1)]
# df_trimmed


CV = 5

# =============================================================================
# scikit-learn: Linear Models
# =============================================================================
# =============================================================================
# X = df_scaled.pipe(get_X)
# for solver in (
#     ElasticNetCV(cv=CV, random_state=RANDOM_STATE),
#     LassoCV(cv=CV, random_state=RANDOM_STATE),
#     RidgeCV(cv=CV)
# ):
#     for regressor in ('depth', 'width'):
#         fit_linear_cv_model(X, regressor, solver)
# =============================================================================


# # =============================================================================
# # Ensembles
# # =============================================================================
# X = df_scaled.pipe(get_X)
# for solver in (
#     AdaBoostRegressor(random_state=RANDOM_STATE),
#     BaggingRegressor(random_state=RANDOM_STATE),
#     ExtraTreesRegressor(random_state=RANDOM_STATE),
#     GradientBoostingRegressor(random_state=RANDOM_STATE),
#     HistGradientBoostingRegressor(random_state=RANDOM_STATE),
#     RandomForestRegressor(random_state=RANDOM_STATE),
# ):
#     for regressor in ('depth', 'width'):
#         y = df_scaled.pipe(get_y, regressor=regressor)
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, train_size=.8
#         )

#         solver.fit(X_train, y_train)

#         mae = mean_absolute_error(y_test, solver.predict(X_test))
#         mse = mean_squared_error(y_test, solver.predict(X_test))
#         r2_ = r2_score(y_test, solver.predict(X_test))

#         print(f'MAE: {mae:,.6f}')
#         print(f'MSE: {mse:,.6f}')
#         print(f'R^2: {r2_:,.6f}')


# =============================================================================
# Examples
# --------
# >>> from sklearn.datasets import load_diabetes
# >>> from sklearn.linear_model import RidgeCV
# >>> from sklearn.svm import LinearSVR
# >>> from sklearn.ensemble import RandomForestRegressor
# >>> from sklearn.ensemble import StackingRegressor
# >>> X, y = load_diabetes(return_X_y=True)
# >>> estimators = [
# ...     ('lr', RidgeCV()),
# ...     ('svr', LinearSVR(random_state=RANDOM_STATE))
# ... ]
# >>> reg = StackingRegressor(
# ...     estimators=estimators,
# ...     final_estimator=RandomForestRegressor(n_estimators=10,
# ...                                           random_state=RANDOM_STATE)
# ... )
# >>> from sklearn.model_selection import train_test_split
# >>> X_train, X_test, y_train, y_test = train_test_split(
# ...     X, y, random_state=RANDOM_STATE
# ... )
# >>> reg.fit(X_train, y_train).score(X_test, y_test)
# 0.3...
# """
# =============================================================================


# =============================================================================
# Examples
# --------
# >>> import numpy as np
# >>> from sklearn.linear_model import LinearRegression
# >>> from sklearn.ensemble import RandomForestRegressor
# >>> from sklearn.ensemble import VotingRegressor
# >>> from sklearn.neighbors import KNeighborsRegressor
# >>> r1 = LinearRegression()
# >>> r2 = RandomForestRegressor(n_estimators=10, random_state=1)
# >>> r3 = KNeighborsRegressor()
# >>> X = np.array([[1, 1], [2, 4], [3, 9], [4, 16], [5, 25], [6, 36]])
# >>> y = np.array([2, 6, 12, 20, 30, 42])
# >>> er = VotingRegressor([('lr', r1), ('rf', r2), ('r3', r3)])
# >>> print(er.fit(X, y).predict(X))
# [ 6.8...  8.4... 12.5... 17.8... 26...  34...]
#
# In the following example, we drop the `'lr'` estimator with
# :meth:`~VotingRegressor.set_params` and fit the remaining two estimators:
#
# >>> er = er.set_params(lr='drop')
# >>> er = er.fit(X, y)
# >>> len(er.estimators_)
# 2
# """
# =============================================================================

# regression = GradientBoostingRegressor(random_state=RANDOM_STATE)
# param_grid = {
#     'n_estimators': [100, 200, 500],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth': [4, 5, 6],
#     'criterion': ['squared_error']
# }
# gscv = GridSearchCV(estimator=regression,
#                     param_grid=param_grid, cv=5, verbose=2)
# gscv.fit(X_train, y_train)
# gscv.best_params_

# =============================================================================
# Create dataset
# =============================================================================
df = get_data_frame()

# =============================================================================
# Create Scaled DataFrame
# =============================================================================
scaler = StandardScaler()
df_scaled = pd.DataFrame(data=scaler.fit_transform(df), columns=df.columns)
X, y = df_scaled.pipe(get_X_y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8)

max_depth = 30
cv = 5
random_state = 42
for solver in (
    ElasticNetCV(cv=cv, random_state=random_state),
    LassoCV(cv=cv, random_state=random_state),
    RidgeCV(cv=cv)
):
    regr_multirf = MultiOutputRegressor(solver)
    regr_multirf.fit(X_train, y_train)
    y_multirf = regr_multirf.predict(X_test)

    # =========================================================================
    # Plot the results
    # =========================================================================
    plt.figure()
    size = 50
    alpha = .4
    plt.scatter(
        y_test[:, 0],
        y_test[:, 1],
        edgecolor='k',
        c='navy',
        s=size,
        marker='s',
        alpha=alpha,
        label='Data',
    )
    plt.scatter(
        y_multirf[:, 0],
        y_multirf[:, 1],
        edgecolor='k',
        c='cornflowerblue',
        s=size,
        alpha=alpha,
        label=f'Multi RF score = {regr_multirf.score(X_test, y_test):,.2f}',
    )
    plt.xlim([-3, 3])
    plt.ylim([-2, 4])
    plt.xlabel('depth')
    plt.ylabel('width')
    plt.title('Comparing random forests and the multi-output meta estimator')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
