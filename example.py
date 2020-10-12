# -*- coding: utf-8 -*-

"""
该程序用于测试l5kit的功能，并研究数据集
"""

import os
import torch
import torch.nn as nn

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer

# 设置基本环境
os.environ["L5KIT_DATA_FOLDER"] = "D:/kaggle/prediction-dataset"
# dm储存了环境路径，可用于获取路径下的内容的地址
dm = LocalDataManager(None)
# 读入基本参数，储存在字典中
cfg = load_config_data('./example.yaml')


def build_model(config):
    """
    该函数根据读入的参数返回模型
    """
    model = None
    return model


def forward(data, model, device, criterion):
    """
    该函数作为前向函数
    """
    loss = None
    output = None
    return loss, output


if __name__ == "__main__":
    train_cfg = cfg['train_data_loader']
    # 生成网格
    rasterizer = build_rasterizer(cfg, dm)
    # 获取包含训练数据集的实例，内部包含agents、frames和scenes
    train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
    frame = train_zarr.frames[0]
    agents = train_zarr.agents[0]
    # 获取数据集内的特定内容
    ego_dataset = EgoDataset(cfg, train_zarr, rasterizer)
    agent_dataset = AgentDataset(cfg, train_zarr, rasterizer)
    ego_data = ego_dataset[0]
    agent_data = agent_dataset[0]
    input()
