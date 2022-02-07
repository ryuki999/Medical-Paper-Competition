from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from models import BaseModel


class Predictor(object):
    """推論器"""

    def __init__(self):
        pass

    def inference(
        self,
        config: Config,
        test_loader: DataLoader,
        model: BaseModel,
    ) -> List[List[float]]:

        preds = []
        for _, (input_ids, attention_mask) in tqdm(
            enumerate(test_loader), total=len(test_loader)
        ):
            input_ids = input_ids.to(config.DEVICE)
            attention_mask = attention_mask.to(config.DEVICE)
            with torch.no_grad():
                y_preds = model(input_ids, attention_mask)
            preds.append(y_preds.to("cpu").numpy())
        preds = np.concatenate(preds)

        return preds
