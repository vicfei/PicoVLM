import torch
import torch.nn.functional as F


def collate_answer_only(samples, pad_id=0, eos_id=1):
    """samples 中每项为 (image, prompt_ids, answer_ids)，answer_ids 不包含 EOS。"""
    # TODO: 返回 images、input_ids、labels；只监督 answer+EOS，其他位置是 -100。
    raise NotImplementedError


def shifted_answer_loss(logits, text_labels, image_token_count):
    """logits 包含图像和文本位置；text_labels 只包含文本位置。"""
    # TODO: 前补 image_token_count 个 -100，然后做 next-token shift 和交叉熵。
    raise NotImplementedError
