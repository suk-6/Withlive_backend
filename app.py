from flask import Flask, render_template, request, redirect, url_for, flash
from predict import predict

app = Flask(__name__)

@app.route('/')
def index():
    return "OK"

@app.route('/api', methods=['POST'])
def api():
    return predict(request.json)


if __name__ == '__main__':
    app.run(debug=True, port=8000, host='0.0.0.0')