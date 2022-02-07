import time
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import fbeta_score
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from config import Config
from models import BaseModel
from utils import AverageMeter, timeSince


class Trainer(object):
    """訓練器"""

    def __init__(self) -> None:
        pass

    def train(
        self,
        fold: int,
        border: float,
        valid_folds: pd.DataFrame,
        config: Config,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        model: BaseModel,
        criterion: _Loss,
        optimizer: Optimizer,
    ) -> pd.DataFrame:
        """与えられたデータにより訓練を行う

        Args:
            fold (int): 現在のfold数
            border (float): このfoldでの1に分類するための閾値
            valid_folds (pd.DataFrame): 現在のfoldでの検証データ
            config (Config): 設定したパラメータ
            train_loader (DataLoader): 訓練データの読み込み器
            valid_loader (DataLoader): 検証データの読み込み器
            model (BaseModel): モデル
            criterion (_Loss): 損失関数
            optimizer (Optimizer): 最適化関数

        Returns:
            pd.DataFrame: 現在のfoldでの検証データに対する予測値
        """
        self.config = config
        # ====================================================
        # Loop
        # ====================================================
        best_score = -1

        for epoch in range(config.EPOCHS):
            start_time = time.time()

            # train
            avg_loss = self._train_fn(train_loader, model, criterion, optimizer, epoch)

            # eval
            avg_val_loss, preds = self._valid_fn(valid_loader, model, criterion)
            valid_labels = valid_folds["judgement"].values

            # scoring
            score = fbeta_score(valid_labels, np.where(preds < border, 0, 1), beta=7.0)

            elapsed = time.time() - start_time
            config.LOGGER.info(
                f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss:{avg_val_loss:.4f} time:{elapsed:.0f}s"
            )
            config.LOGGER.info(f"Epoch {epoch+1} - Score: {score}")

            if score > best_score:
                best_score = score
                config.LOGGER.info(
                    f"Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model"
                )
                torch.save(
                    {"model": model.state_dict(), "preds": preds},
                    config.OUTPUT_DIR + f"bert-base-uncased_fold{fold}_best.pth",
                )

        check_point = torch.load(
            config.OUTPUT_DIR + f"bert-base-uncased_fold{fold}_best.pth"
        )

        valid_folds["preds"] = check_point["preds"]

        return valid_folds

    def _train_fn(
        self,
        train_loader: DataLoader,
        model: BaseModel,
        criterion: _Loss,
        optimizer: Optimizer,
        epoch: int,
    ) -> Optional[Any]:
        start = time.time()
        losses = AverageMeter()

        # switch to train mode
        model.train()

        for step, (input_ids, attention_mask, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = input_ids.to(self.config.DEVICE)
            attention_mask = attention_mask.to(self.config.DEVICE)
            labels = labels.to(self.config.DEVICE)
            batch_size = labels.size(0)

            y_preds = model(input_ids, attention_mask)

            loss = criterion(y_preds, labels)

            # record loss
            losses.update(loss.item(), batch_size)
            loss.backward()

            optimizer.step()

            if step % 100 == 0 or step == (len(train_loader) - 1):
                print(
                    f"Epoch: [{epoch + 1}][{step}/{len(train_loader)}] "
                    f"Elapsed {timeSince(start, float(step + 1) / len(train_loader)):s} "
                    f"Loss: {losses.avg:.4f} "
                )

        return losses.avg

    def _valid_fn(
        self, valid_loader: DataLoader, model: BaseModel, criterion: _Loss
    ) -> Tuple[Optional[Any], pd.DataFrame]:
        start = time.time()
        losses = AverageMeter()

        # switch to evaluation mode
        model.eval()
        preds = []

        for step, (input_ids, attention_mask, labels) in enumerate(valid_loader):
            input_ids = input_ids.to(self.config.DEVICE)
            attention_mask = attention_mask.to(self.config.DEVICE)
            labels = labels.to(self.config.DEVICE)
            batch_size = labels.size(0)

            # compute loss
            with torch.no_grad():
                y_preds = model(input_ids, attention_mask)

            loss = criterion(y_preds, labels)
            losses.update(loss.item(), batch_size)

            # record score
            preds.append(y_preds.to("cpu").numpy())

            if step % 100 == 0 or step == (len(valid_loader) - 1):
                print(
                    f"EVAL: [{step}/{len(valid_loader)}] "
                    f"Elapsed {timeSince(start, float(step + 1) / len(valid_loader)):s} "
                    f"Loss: {losses.avg:.4f} "
                )

        predictions = np.concatenate(preds)
        return losses.avg, predictions
