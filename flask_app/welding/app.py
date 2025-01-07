import sys

import numpy as np
from flask import Flask, render_template, request

SRC_DIR = '/home/green-machine/data_science/bmstu_graduate_project/src'
sys.path.insert(1, SRC_DIR)

from features.build_features import get_X_y_scalers
from models.predict_model import load_trained_model

app = Flask(__name__)


def init():
    return (*get_X_y_scalers(), load_trained_model())


@app.route('/', methods=['POST', 'GET'])
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
