import torch


@torch.no_grad()
def greedy_generate(model, images, prompt_ids, eos_id=1, max_new_tokens=16):
    was_training = model.training
    model.eval()
    generated = list(prompt_ids)
    try:
        for _ in range(max_new_tokens):
            input_ids = torch.tensor([generated], dtype=torch.long, device=images.device)
            output = model(images, input_ids)
            logits = output[0] if isinstance(output, tuple) else output
            next_id = logits[0, -1].argmax().item()
            generated.append(next_id)
            if next_id == eos_id:
                break
    finally:
        model.train(was_training)
    return generated[len(prompt_ids):]
