#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 22:06:59 2025

@author: alexandermikhailov
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.multioutput import MultiOutputRegressor

from welding_ml.welding_ml._graduate_thesis import maes, mses, r2_s


def plot_model_train_val_losses(history_dict: dict[str, list[float]]) -> None:
    """
    Plots Train & Validation Losses per Epoch
    """
    plt.figure(figsize=(8, 5))
    for array in history_dict.values():
        plt.plot(array)

    plt.title('Loss Plot')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
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

    caption = solver.get_params()['estimator'] if isinstance(
        solver, MultiOutputRegressor
    ) else type(solver).__name__

    plt.figure()
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
        y_pred[:, 0],
        y_pred[:, 1],
        edgecolor='k',
        c='cornflowerblue',
        s=size,
        alpha=alpha,
        label=(
            f'MAE: {maes[caption]:,.6f}; '
            f'MSE: {mses[caption]:,.6f}; '
            f'$R^2$: {r2_s[caption]:,.6f}'
        )
    )
    plt.xlim([-3, 3])
    plt.ylim([-2, 4])
    plt.xlabel('depth')
    plt.ylabel('width')
    plt.title(caption)
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
