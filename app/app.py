#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 21:52:45 2025

@author: alexandermikhailov
"""

from flask import Flask, render_template, request
from welding_ml.config import DIMENSIONS
from welding_ml.features import get_X_y_scalers
from welding_ml.modeling.predict import (load_trained_model,
                                         make_prediction_scaled)

from app.config.config import get_config_by_name


def create_app(config=None) -> Flask:
    """Create a Flask application.

    Args:
        config (_type_, optional): The configuration object to use. Defaults to None.

    Returns:
        Flask: A Flask application instance.
    """

    app = Flask(__name__)

    if config:
        app.config.from_object(get_config_by_name(config))

    model = load_trained_model()

    scaler_X, scaler_y = get_X_y_scalers()

    @app.route('/', methods=['GET', 'POST'])
    def main():
        if request.method == 'GET':
            return render_template('index.html')

        if request.method == 'POST':
            input_features = request.form['input_features']

            y_pred = make_prediction_scaled(
                input_features,
                model,
                scaler_X,
                scaler_y
            ).flatten()

            data = dict(
                zip(
                    DIMENSIONS,
                    map(lambda _: f'{_:,.6f}', y_pred.tolist())
                )
            )

            return render_template('index.html', result=data)

    return app
