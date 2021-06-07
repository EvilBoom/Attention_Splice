# _*_ coding: utf-8 _*_
# @Time : 2021/6/6 12:08
# @Author : 张宝宇
# @Version：V 0.0
# @File : p_attention.py
# @desc :
import argparse
import random

import numpy as np
import torch

from configs import *
from att_frame import Att_Frame


def seed_torch(m_seed=2021):
    random.seed(m_seed)
    np.random.seed(m_seed)
    torch.manual_seed(m_seed)


if __name__ == '__main__':
    # 设置参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(seed)
    # frame
    framework = Att_Frame(batch_size, lr, epoch)
    framework.train_start()
