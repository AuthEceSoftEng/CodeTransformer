import json

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from serve import api

app = Flask(__name__)
CORS(app)
predict = api()

@app.route('/')
def root():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    inputData = request.json
    outputData = predict(inputData)
    response = jsonify(outputData)

    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)