import torch


def train_step(model, batch, optimizer, max_grad_norm=1.0):
    model.train()
    optimizer.zero_grad()
    loss = model(*batch)
    if loss.ndim != 0:
        raise ValueError("model 必须返回标量 loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        max_grad_norm,
    )
    optimizer.step()
    return loss.detach().item()
