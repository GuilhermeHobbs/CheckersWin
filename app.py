from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

@app.route('/ask-name')
def ask_name():
    return "It's me"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
