from flask import Flask, request
from flask_cors import CORS
import sys

#import numpy as np
#import torch
#import torch.nn as nn
#from torch.nn import functional as F

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
  
@app.route('/move')
def ask_name():
    app.logger.info("Eiiiii")  # Use Flask's logger instead of print
    logger.info("Eiiiii 2")  # This should show in Render logs
    print("Eiiiii 3", flush=True)  # Force immediate flushing
    sys.stdout.flush()  # Additional explicit flush
  
    return "HELLO"
  
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)  # Render requires explicit host/port
