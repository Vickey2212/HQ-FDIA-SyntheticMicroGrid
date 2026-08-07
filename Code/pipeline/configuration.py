# =========================
# EXPERIMENT CONFIGURATION
# =========================

SEED = 7

BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-4

PATIENCE = 4
MIN_DELTA = 1e-4

# Temporal module
T_OUT = 32

# Graph module
NODE_HIDDEN = 64
GNN_STEPS = 4

# Classification threshold
THRESHOLD = 0.45

# IEEE 30-bus network topology.
# Bus numbering follows the standard 1-based IEEE convention.
EDGE_LIST_1BASED = [
    (1, 2),
    (1, 3),
    ...
    (25, 26),
]
