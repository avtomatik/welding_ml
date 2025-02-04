#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 21:52:45 2025

@author: alexandermikhailov
"""

# =============================================================================
# Suggested:
# =============================================================================
# =============================================================================
# from flask import Flask
#
# from app.config.config import get_config_by_name
# from app.initialize_functions import (initialize_db, initialize_route,
#                                       initialize_swagger)
#
#
# def create_app(config=None) -> Flask:
#     """
#     Create a Flask application.
#
#     Args:
#         config: The configuration object to use.
#
#     Returns:
#         A Flask application instance.
#     """
#     app = Flask(__name__)
#     if config:
#         app.config.from_object(get_config_by_name(config))
#
#     # Initialize extensions
#     initialize_db(app)
#
#     # Register blueprints
#     initialize_route(app)
#
#     # Initialize Swagger
#     initialize_swagger(app)
#
#     return app
# =============================================================================


import numpy as np
from flask import Flask, render_template, request

from welding_ml.features import get_X_y_scalers
from welding_ml.modeling.predict import load_trained_model

app = Flask(__name__)


def init():
    return (*get_X_y_scalers(), load_trained_model())


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        DIMENSIONS = ('Depth', 'Width')

        scaler_X, scaler_y, clf = init()

        input_params = request.form['input_params']
        model_input = scaler_X.transform(
            np.array(list(map(float, input_params.split()))).reshape(1, -1)
        )
        y_pred = scaler_y.inverse_transform(clf.predict(model_input))

        result = [
            dict(zip(DIMENSIONS, map(lambda _: f'{_:,.6f}', row)))
            for row in y_pred.tolist()
        ]

        return render_template('index.html', result=result[0])


if __name__ == '__main__':
    app.run(debug=True)
