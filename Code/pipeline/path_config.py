from pathlib import Path

# =========================
# PATH CONFIGURATION
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

TRAIN_CSV_PATH = DATA_DIR / "train_selected.csv"
VAL_CSV_PATH = DATA_DIR / "val_selected.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_PATH = OUTPUT_DIR / "best_classical_end2end_binary.pt"

# =========================
# ARGUMENT PARSER
# =========================

parser = argparse.ArgumentParser(
    description="Train the classical end-to-end binary FDIA detection model."
)

parser.add_argument(
    "--train",
    type=str,
    required=True,
    help="Path to the training CSV file."
)

parser.add_argument(
    "--val",
    type=str,
    required=True,
    help="Path to the validation CSV file."
)

parser.add_argument(
    "--output",
    type=str,
    default="results/models",
    help="Directory for saving the best model."
)

args = parser.parse_args()

TRAIN_CSV_PATH = Path(args.train)
VAL_CSV_PATH = Path(args.val)

OUTPUT_DIR = Path(args.output)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAVE_DIR = Path(args.output)
SAVE_DIR.mkdir(parents=True, exist_ok=True) = Path(args.output)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_PATH = OUTPUT_DIR / "best_classical_end2end_binary.pt"
