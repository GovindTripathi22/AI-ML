"""
Pipeline Configuration for Business Entity Resolution.

Defines default hyperparameters, directory locations, and model thresholds.
"""

from pathlib import Path

# ---------------------------------------------------------
# Configurable Constants
# ---------------------------------------------------------
BLOCKING_TOP_K = 8
MATCH_THRESHOLD = 0.60
RANDOM_SEED = 42
USE_EMBEDDINGS = False

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
OUTPUT_DIR = "output"


# ---------------------------------------------------------
# Resolved Path References
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / TRAIN_DIR
TEST_PATH = PROJECT_ROOT / TEST_DIR
OUTPUT_PATH = PROJECT_ROOT / OUTPUT_DIR
