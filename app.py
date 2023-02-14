import pickle

import numpy as np
from flask import Flask, request, jsonify

model = pickle.load(open(
    'C:\\Users\\user\\OneDrive - Teesside University\\Documents\\TEESSIDE UNI\\YR 3 first semester\\Final Year Project\\model.pkl',
    'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return "Hello world"


@app.route('/predict', methods=['POST'])
def predict():
    age = request.form.get('age')
    cp = request.form.get('cp')
    trestbps = request.form.get('trestbps')
    chol = request.form.get('chol')
    thalach = request.form.get('thalach')
    oldpeak = request.form.get('oldpeak')
    slope = request.form.get('slope')
    ca = request.form.get('ca')
    thal = request.form.get('thal')
    sex = request.form.get('sex')
    fbs = request.form.get('fbs')
    restecg = request.form.get('restecg')
    exang = request.form.get('exang')

    input_query = np.array([[age, cp, trestbps, chol, thalach, oldpeak, slope, ca, thal, sex, fbs, restecg, exang]], dtype= 'i')

    target = model.predict(input_query)[0]


    return jsonify({'heart_disease': str(target)})

if __name__ == '__main__':
    app.run(debug=True)