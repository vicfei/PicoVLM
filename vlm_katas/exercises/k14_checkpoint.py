import torch


def save_checkpoint(path, model, tokenizer, config):
    """保存模型、完整 tokenizer 词表和模型配置。"""
    # TODO
    raise NotImplementedError


def load_checkpoint(path, model, map_location="cpu"):
    """加载权重，并返回包含 tokenizer/config 的元数据。"""
    # TODO
    raise NotImplementedError
