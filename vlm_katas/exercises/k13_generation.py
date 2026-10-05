import torch


@torch.no_grad()
def greedy_generate(model, images, prompt_ids, eos_id=1, max_new_tokens=16):
    """单样本 greedy decoding，返回新生成的 token，不包含 prompt。"""
    # TODO: 临时切到 eval，逐 token 生成，遇 EOS 停止，并恢复原训练状态。
    raise NotImplementedError
