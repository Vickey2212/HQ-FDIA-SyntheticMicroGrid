!pip install pennylane
!pip install -U pennylane pennylane-lightning[gpu]
import re
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pennylane as qml.
from tqdm.auto import tqdm
