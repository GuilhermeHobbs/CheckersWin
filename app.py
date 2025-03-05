from flask import Flask, request
from flask_cors import CORS

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

temp = torch.rand([1, 3, 256])
print("SHAPE:",temp.shape)
lala = nn.LayerNorm(256)
print(lala(temp))


app = Flask(__name__)
CORS(app)  # This enables CORS for all routes


  
@app.route('/move')
def ask_name():
    
    temp = torch.rand([1, 3, 256])
    print("SHAPEEE:",temp.shape)
    lala = nn.LayerNorm(256)
    print(lala(temp))
    


