import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers as T
from sklearn.metrics import fbeta_score
from torch.utils.data import DataLoader

from config import Config
from models import BaseModel
from predictor import Predictor
from preprocessing import BaseDataset, get_test_data, get_train_data
from trainer import Trainer
from utils import seed_torch

warnings.filterwarnings("ignore")


class MedicalPaperClassifier(object):
    """医学論文分類器"""

    def __init__(self) -> None:
        pass

    def _get_result(self, result_df: pd.DataFrame, border: float):
        preds = result_df["preds"].values
        labels = result_df["judgement"].values
        score = fbeta_score(labels, np.where(preds < border, 0, 1), beta=7.0)
        return score

    def train_loop(
        self, config: Config, train: pd.DataFrame, trainer: Trainer
    ) -> pd.DataFrame:
        """医学論文分類器の訓練を行う

        Args:
            config (Config): 設定パラメータ
            train (pd.DataFrame): 訓練データ
            trainer (Trainer): 訓練器

        Returns:
            pd.DataFrame: 検証データに対する予測値
        """
        # この値を境に、CV値を算出する
        border = len(train[train["judgement"] == 1]) / len(train["judgement"])
        # print(border)
        oof_df = pd.DataFrame()
        for fold in range(config.FOLD):

            config.LOGGER.info(f"========== fold: {fold} training ==========")

            # ===========
            # Data Loader
            # ===========
            trn_idx = train[train["fold"] != fold].index
            val_idx = train[train["fold"] == fold].index

            train_folds = train.loc[trn_idx].reset_index(drop=True)
            valid_folds = train.loc[val_idx].reset_index(drop=True)

            train_dataset = BaseDataset(
                train_folds, config.MODEL_NAME, config.MAX_LENGTH
            )
            valid_dataset = BaseDataset(
                valid_folds, config.MODEL_NAME, config.MAX_LENGTH
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True,
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                drop_last=False,
            )

            # =====
            # Model
            # =====
            model = BaseModel(config.MODEL_NAME)
            model.to(config.DEVICE)

            _oof_df = trainer.train(
                config=config,
                fold=fold,
                border=border,
                valid_folds=valid_folds,
                train_loader=train_loader,
                valid_loader=valid_loader,
                model=model,
                criterion=nn.BCELoss(),
                optimizer=T.AdamW(model.parameters(), lr=config.REARNING_RATE),
            )

            oof_df = pd.concat([oof_df, _oof_df])
            config.LOGGER.info(f"========== fold: {fold} result ==========")
            score = self._get_result(_oof_df, border)
            config.LOGGER.info(f"Score: {score:<.5f}")

        # CV result
        config.LOGGER.info("========== CV ==========")
        score = self._get_result(oof_df, border)
        config.LOGGER.info(f"CV Score: {score:<.5f}")

        return oof_df

    def inferrence_loop(
        self, config: Config, test: pd.DataFrame, predictor: Predictor
    ) -> pd.DataFrame:
        """医学論文分類器の推論を行う

        Args:
            config (Config): 設定パラメータ
            test (pd.DataFrame): テストデータ
            predictor (Predictor): 予測器

        Returns:
            pd.DataFrame: テストデータに対する予測値
        """
        predictions = []

        test_dataset = BaseDataset(
            test, config.MODEL_NAME, config.MAX_LENGTH, include_labels=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True
        )

        for i in range(config.FOLD):

            config.LOGGER.info(
                f"========== model: bert-base-uncased fold: {i} inference =========="
            )
            model = BaseModel(config.MODEL_NAME)
            model.to(config.DEVICE)
            model.load_state_dict(
                torch.load(config.OUTPUT_DIR + f"bert-base-uncased_fold{i}_best.pth")[
                    "model"
                ]
            )
            model.eval()

            # Inference
            preds = predictor.inference(
                config=config,
                test_loader=test_loader,
                model=model,
            )
            predictions.append(preds)
        predictions = np.mean(predictions, axis=0)

        # pd.Series(predictions).to_csv(
        #     config.OUTPUT_DIR + "predictions.csv", index=False
        # )
        predictions1 = np.where(predictions < 0.0262, 0, 1)

        return predictions1


def train(train_path: Path):
    """医学論文分類器の訓練を行う

    Args:
        train_path (Path): 訓練データまでのPath
    """

    train = get_train_data(train_path, n_splits=Config.FOLD, random_state=Config.SEED)

    classifier = MedicalPaperClassifier()
    oof_df = classifier.train_loop(Config, train, trainer=Trainer())
    # Save OOF result
    oof_df.to_csv(Config.OUTPUT_DIR + "oof_df.csv", index=False)


def inferrence(test_path: Path, sample_sub_path: Path):
    """医学論文分類器による推論を行う

    Args:
        test_path (Path): テストデータまでのPath
        sample_sub_path (Path): sample_submitまでのPath
    """
    sub = pd.read_csv(sample_sub_path, header=None)
    sub.columns = ["id", "judgement"]
    test = get_test_data(test_path)

    classifier = MedicalPaperClassifier()
    predictions = classifier.inferrence_loop(Config, test, predictor=Predictor())

    # submission
    sub["judgement"] = predictions
    sub.to_csv(Config.OUTPUT_DIR + "submission.csv", index=False, header=False)


if __name__ in "__main__":
    seed_torch(Config.SEED)

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", default="train.csv", required=True)
    parser.add_argument("--test_data", default="test.csv", required=True)
    parser.add_argument("--sample_submit", default="sample_submit.csv", required=True)

    args = parser.parse_args()

    train(Path(Config.DATA_DIR + args.train_data))
    inferrence(
        Path(Config.DATA_DIR + args.test_data),
        Path(Config.DATA_DIR + args.sample_submit),
    )
