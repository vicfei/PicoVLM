import torch
import torch.nn.functional as F


def collate_answer_only(samples, pad_id=0, eos_id=1):
    images = torch.stack([sample[0] for sample in samples])
    sequences = [prompt + answer + [eos_id] for _, prompt, answer in samples]
    max_length = max(map(len, sequences))
    input_ids = torch.full((len(samples), max_length), pad_id, dtype=torch.long)
    labels = torch.full((len(samples), max_length), -100, dtype=torch.long)
    for row, (_, prompt, answer) in enumerate(samples):
        answer_with_eos = answer + [eos_id]
        input_ids[row, :len(sequences[row])] = torch.tensor(sequences[row])
        labels[row, len(prompt):len(prompt) + len(answer_with_eos)] = torch.tensor(answer_with_eos)
    return images, input_ids, labels


def shifted_answer_loss(logits, text_labels, image_token_count):
    if logits.size(1) != text_labels.size(1) + image_token_count:
        raise ValueError("logits 长度必须等于 image tokens + text labels")
    ignored_images = torch.full(
        (text_labels.size(0), image_token_count), -100,
        dtype=torch.long, device=text_labels.device,
    )
    full_labels = torch.cat((ignored_images, text_labels), dim=1)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        full_labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
