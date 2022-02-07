import math
import os
import random
import time
from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger

import numpy as np
import torch


def init_logger(log_file):
    """Loggerの定義"""

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def seed_torch(seed):
    """seed値の固定"""
    # python の組み込み関数の seed を固定
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # numpy の seed を固定
    np.random.seed(seed)
    # torch の seed を固定
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 決定論的アルゴリズムを使用する
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """平均・現在の値を計算し保存する"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))
