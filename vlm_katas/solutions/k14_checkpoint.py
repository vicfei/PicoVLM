import torch


def save_checkpoint(path, model, tokenizer, config):
    payload = {
        "model_state": model.state_dict(),
        "tokenizer": {"itos": list(tokenizer.itos), "stoi": dict(tokenizer.stoi)},
        "config": dict(config),
    }
    torch.save(payload, path)


def load_checkpoint(path, model, map_location="cpu"):
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # 兼容较旧的 PyTorch
        payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state"])
    return {"tokenizer": payload["tokenizer"], "config": payload["config"]}
