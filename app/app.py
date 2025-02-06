#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 21:52:45 2025

@author: alexandermikhailov
"""

import numpy as np
from flask import Flask, render_template, request
from welding_ml.config import DIMENSIONS
from welding_ml.features import get_X_y_scalers
from welding_ml.modeling.predict import load_trained_model

from app.config.config import get_config_by_name
# from app.initialize_functions import (initialize_db, initialize_route,
#                                       initialize_swagger)

from .services import validate_input


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

# =============================================================================
# Initialize extensions
# =============================================================================
    # initialize_db(app)

# =============================================================================
# Register blueprints
# =============================================================================
    # initialize_route(app)

# =============================================================================
# Initialize Swagger
# =============================================================================
    # initialize_swagger(app)

    scaler_X, scaler_y = get_X_y_scalers()

    model = load_trained_model()

    @app.route('/', methods=['POST'])
    def main():
        input_features = request.form['input_features']

        # =====================================================================
        # TODO: Extract `make_prediction` Function
        # =====================================================================

        input_features_scaled = scaler_X.transform(
            np.array(validate_input(input_features)).reshape(1, -1)
        )
        y_pred = scaler_y.inverse_transform(
            model.predict(input_features_scaled)
        )

        result = [
            dict(zip(DIMENSIONS, map(lambda _: f'{_:,.6f}', row)))
            for row in y_pred.tolist()
        ]

        return render_template('index.html', result=result[0])

    return app
