from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

@app.route('/move')
def ask_name():
    a = request.args.get('a', '')
    b = request.args.get('b', '')
    
    return a+b
   
