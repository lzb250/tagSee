# src/config.py

MODEL_PATH = "models/bert-base-chinese"
SAVE_PATH = "models/skill_extraction_model"

LABEL_LIST = ["O", "B-SKILL", "I-SKILL"]

MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
