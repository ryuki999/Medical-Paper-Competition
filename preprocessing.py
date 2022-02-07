from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class BaseDataset(Dataset):
    """BERT学習のためのBaseDataset"""

    def __init__(self, df, model_name, max_length, include_labels=True):
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.df = df
        self.include_labels = include_labels

        df["title_abstract"] = df["title"] + " " + df["abstract"].fillna("")
        sentences = df["title_abstract"].tolist()

        max_length = max_length
        self.encoded = tokenizer.batch_encode_plus(
            sentences,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
        )

        if self.include_labels:
            self.labels = df["judgement"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.encoded["input_ids"][idx])
        attention_mask = torch.tensor(self.encoded["attention_mask"][idx])

        if self.include_labels:
            label = torch.tensor(self.labels[idx]).float()
            return input_ids, attention_mask, label

        return input_ids, attention_mask


def get_train_data(train_path: Path, n_splits: int, random_state: int) -> pd.DataFrame:
    """訓練データに前処理を行う

    Args:
        train_path (Path): 訓練データまでのPath
        n_splits (int): 交差検証の分割数
        random_state (int): seed値

    Returns:
        pd.DataFrame: 前処理後の訓練データ
    """
    train = pd.read_csv(train_path).iloc[:500, :]

    # id=2488: judgement=1となっているが、正しくはjudgement=0
    # id=7708: judgement=1となっているが、正しくはjudgement=0
    train["judgement"][2488] = 0
    train["judgement"][7708] = 0
    fold = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
    for n, (_, val_index) in enumerate(fold.split(train, train["judgement"])):
        train.loc[val_index, "fold"] = int(n)
    train["fold"] = train["fold"].astype(np.uint8)

    return train


def get_test_data(test_path: Path) -> pd.DataFrame:
    """テストデータに前処理を行う

    Args:
        test_path (Path): テストデータまでのPath

    Returns:
        pd.DataFrame: 前処理後のテストデータ
    """
    test = pd.read_csv(test_path)

    return test
