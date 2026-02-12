# train.py

from src.dataset_generator import generate_dataset
from src.ner_dataset_builder import build_ner_dataset
from src.train_ner import train

if __name__ == "__main__":
    generate_dataset(1000)
    build_ner_dataset()
    train()
