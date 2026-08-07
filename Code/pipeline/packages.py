!pip install pennylane
!pip install -U pennylane pennylane-lightning[gpu]
import re
import numpy as np
import pandas as pd
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import seaborn as sns
import time
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import train_test_split
from collections import Counter
import pennylane as qml
import random
from tqdm.auto import tqdm
