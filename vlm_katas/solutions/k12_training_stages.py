VALID_STAGES = {"vision_pretrain", "language_pretrain", "align", "sft"}


def configure_stage(model, stage: str):
    if stage not in VALID_STAGES:
        raise ValueError(f"未知阶段: {stage}")
    enabled = {
        "vision_pretrain": {"vision"},
        "language_pretrain": {"llm"},
        "align": {"projector"},
        "sft": {"projector", "llm"},
    }[stage]
    for name in ("vision", "projector", "llm"):
        module = getattr(model, name)
        should_train = name in enabled
        module.train(should_train)
        for parameter in module.parameters():
            parameter.requires_grad = should_train


def trainable_parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
