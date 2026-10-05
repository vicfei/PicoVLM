import torch
import torch.nn as nn


class MiniVLMCore(nn.Module):
    def __init__(self, vision, projector, llm):
        super().__init__()
        self.vision = vision
        self.projector = projector
        self.llm = llm

    def splice(self, images, input_ids):
        image_embeddings = self.projector(self.vision.encode(images))
        text_embeddings = self.llm.token_embed(input_ids)
        if image_embeddings.size(0) != text_embeddings.size(0):
            raise ValueError("图像与文本 batch size 不一致")
        if image_embeddings.size(-1) != text_embeddings.size(-1):
            raise ValueError("projector 输出维度必须等于语言 embedding 维度")
        return torch.cat((image_embeddings, text_embeddings), dim=1)

    def forward(self, images, input_ids):
        return self.llm(inputs_embeds=self.splice(images, input_ids))
