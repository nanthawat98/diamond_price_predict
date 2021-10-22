import pickle
import numpy as np

from flask import Flask, render_template, request

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def getPredict():
    x1 = request.form['x1']
    x2 = request.form['x2']
    x3 = request.form['x3']
    XTest = np.array([[x1, x2, x3]], dtype = np.float64)

    predicted = model.predict(XTest)[0]
    return render_template('index.html',
    prediction_text = f'Predicted Price: {predicted}')

if __name__ == '__main__':
    app.run(debug = True)