from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

@app.route('/ask-name')
def ask_name():
    question = request.args.get('question', '')
    if question == "The Name?":
        return "It's me"
    else:
        return "Invalid question"
