PAD_ID, EOS_ID, UNK_ID = 0, 1, 2


class CharTokenizer:
    def __init__(self, corpus):
        chars = sorted(set("".join(corpus)))
        self.itos = ["<pad>", "<eos>", "<unk>"] + chars
        self.stoi = {token: i for i, token in enumerate(self.itos)}

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, text):
        return [self.stoi.get(char, UNK_ID) for char in text]

    def decode(self, ids):
        chars = []
        for token_id in ids:
            if token_id == EOS_ID:
                break
            if token_id != PAD_ID:
                chars.append(self.itos[token_id])
        return "".join(chars)
