#from flask import Flask, request
#from flask_cors import CORS

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


temp = torch.rand([1, 3, 10])
print("SHAAAPE:",x.shape,temp.shape)
lala = nn.LayerNorm(10)
print(lala(temp))
