import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# CONFIG
# -------------------------
CSV_PATH = PROJECT_ROOT / "data"

LABEL_COL = "label_binary" 

# Number of top features to keep
TOP_K = 40

# Split settings
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Output files
OUT_DIR = PROJECT_ROOT / "data"/ "feature_selection_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUT = os.path.join(OUT_DIR, "train_selected.csv")
VAL_OUT = os.path.join(OUT_DIR, "val_selected.csv")
FULL_OUT = os.path.join(OUT_DIR, "full_selected.csv")
FEATURES_JSON = os.path.join(OUT_DIR, "selected_features.json")
RANKING_CSV = os.path.join(OUT_DIR, "feature_ranking.csv")
