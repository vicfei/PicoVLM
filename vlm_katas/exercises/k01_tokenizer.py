PAD_ID, EOS_ID, UNK_ID = 0, 1, 2


class CharTokenizer:
    """字符级 tokenizer。特殊 token 的 id 必须固定为 0、1、2。"""

    def __init__(self, corpus):
        # TODO: 从 corpus 收集去重字符，并建立 stoi/itos。
        raise NotImplementedError

    @property
    def vocab_size(self):
        # TODO
        raise NotImplementedError

    def encode(self, text):
        # TODO: 未知字符映射到 UNK_ID。
        raise NotImplementedError

    def decode(self, ids):
        # TODO: 忽略 PAD，在 EOS 处停止。
        raise NotImplementedError
