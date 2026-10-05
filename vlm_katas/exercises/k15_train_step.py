import torch


def train_step(model, batch, optimizer, max_grad_norm=1.0):
    """完成一个训练 batch；model(*batch) 应直接返回标量 loss。"""
    # TODO: train -> zero_grad -> forward -> backward -> clip -> step。
    # 返回普通 Python float。
    raise NotImplementedError
