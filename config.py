import torch

from utils import init_logger


class Config:
    DATA_DIR = "./data/"
    OUTPUT_DIR = "./"
    LOGGER = init_logger(log_file=OUTPUT_DIR + "train.log")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    FOLD = 3  # 10
    SEED = 471
    EPOCHS = 5
    REARNING_RATE = 2e-5
    BATCH_SIZE = 16
    MAX_LENGTH = 128  # 512
    MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
