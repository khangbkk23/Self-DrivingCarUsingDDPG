import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd


class Encoder(nn.Module):
    