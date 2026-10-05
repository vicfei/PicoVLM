VALID_STAGES = {"vision_pretrain", "language_pretrain", "align", "sft"}


def configure_stage(model, stage: str):
    """按训练阶段设置 vision/projector/llm 的 requires_grad 与 train/eval 状态。"""
    # TODO:
    # vision_pretrain -> 只训练 vision
    # language_pretrain -> 只训练 llm
    # align -> 只训练 projector
    # sft -> 训练 projector + llm
    raise NotImplementedError


def trainable_parameter_count(model):
    # TODO
    raise NotImplementedError
