import torch
import torch.nn as nn


class MiniVLMCore(nn.Module):
    """将 vision、projector、LLM 三个已经构造好的模块组装起来。"""

    def __init__(self, vision, projector, llm):
        super().__init__()
        self.vision = vision
        self.projector = projector
        self.llm = llm

    def splice(self, images, input_ids):
        # TODO: vision.encode -> projector，并与 llm.token_embed 的输出按序列维拼接。
        raise NotImplementedError

    def forward(self, images, input_ids):
        # TODO: 将拼接结果通过 inputs_embeds 送进 LLM。
        raise NotImplementedError
