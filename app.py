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
    print("Eiiiii")
    
    return "HELLO"

