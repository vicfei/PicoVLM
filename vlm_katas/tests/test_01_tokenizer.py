from tests._loader import load

m = load("k01_tokenizer")


def test_special_ids_and_deterministic_vocab():
    tok = m.CharTokenizer(["数字", "字是7"])
    assert tok.itos[:3] == ["<pad>", "<eos>", "<unk>"]
    assert tok.stoi["<pad>"] == m.PAD_ID == 0
    assert tok.stoi["<eos>"] == m.EOS_ID == 1
    assert tok.vocab_size == len(set("数字字是7")) + 3


def test_round_trip_unknown_padding_and_eos():
    tok = m.CharTokenizer(["数字7"])
    ids = tok.encode("数字7")
    assert tok.decode(ids) == "数字7"
    assert tok.encode("X") == [m.UNK_ID]
    assert tok.decode([m.PAD_ID, *ids, m.EOS_ID, ids[0]]) == "数字7"
