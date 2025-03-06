from flask import Flask, request
from flask_cors import CORS

#import numpy as np
#import torch
#import torch.nn as nn
#from torch.nn import functional as F

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

  
@app.route('/move')
def ask_name():
    app.logger.info("Eiiiii")  # Use Flask's logger instead of print
  
    return "HELLO"
  
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)  # Render requires explicit host/port
