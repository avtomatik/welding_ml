"""
============================================================
Comparing random forests and the multi-output meta estimator
============================================================

An example to compare multi-output regression with random forest and
the :ref:`multioutput.MultiOutputRegressor <multiclass>` meta-estimator.

This example illustrates the use of the
:ref:`multioutput.MultiOutputRegressor <multiclass>` meta-estimator
to perform multi-output regression. A random forest regressor is used,
which supports multi-output regression natively, so the results can be
compared.

The random forest regressor will only ever predict values within the
range of observations or closer to zero for each of the targets. As a
result the predictions are biased towards the centre of the circle.

Using a single underlying feature the model learns both the
x and y coordinate as output.

"""

# Author: Tim Head <betatim@gmail.com>
#
# License: BSD 3 clause

from functools import cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


@cache
def get_data_frame() -> pd.DataFrame:
    return pd.read_csv('../data/ebw_data.csv')


def get_X_y(df: pd.DataFrame) -> tuple[np.ndarray]:
    return df.iloc[:, :4].values, df.iloc[:, 4:].values


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
regr_multirf = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=100, max_depth=max_depth, random_state=0)
)
regr_multirf.fit(X_train, y_train)

regr_rf = RandomForestRegressor(
    n_estimators=100, max_depth=max_depth, random_state=2)
regr_rf.fit(X_train, y_train)

# =============================================================================
# Predict on new data
# =============================================================================
y_multirf = regr_multirf.predict(X_test)
y_rf = regr_rf.predict(X_test)

# =============================================================================
# Plot the results
# =============================================================================
plt.figure()
size = 50
alpha = 0.4
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
plt.scatter(
    y_rf[:, 0],
    y_rf[:, 1],
    edgecolor='k',
    c='c',
    s=size,
    marker='^',
    alpha=alpha,
    label=f'RF score = {regr_rf.score(X_test, y_test):,.2f}',
)
plt.xlim([-3, 3])
plt.ylim([-2, 4])
plt.title('Comparing random forests and the multi-output meta estimator')
plt.xlabel('depth')
plt.ylabel('width')
plt.legend(loc='upper left')
plt.grid()
plt.show()
