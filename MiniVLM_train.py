# -*- coding: utf-8 -*-
"""迷你VLM 分阶段训练脚本：阶段0预训练 -> 阶段1投影对齐 -> 阶段2指令微调 -> 评估"""
import time, random, json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from mini_vlm import (CharTokenizer, MiniVLM, PAD_ID, EOS_ID, DEVICE,
                      IMG_TOKEN_NUM)

torch.manual_seed(42); random.seed(42)
CKPT = "./ckpt"
import os; os.makedirs(CKPT, exist_ok=True)

# ---------------- 文本模板（训练数据的"配方"） ----------------
CAPTION_PROMPT = "描述这张图片："
CAPTION_TMPL = "图中的数字是{d}。"
VQA_PAIRS = [
    ("图中是什么数字？", "数字是{d}。"),
    ("这张图片里的手写数字是几？", "是{d}。"),
    ("请识别图中的手写数字：", "{d}。"),
]
corpus = ([CAPTION_PROMPT, CAPTION_TMPL.replace("{d}", "")]
          + [q for q, _ in VQA_PAIRS] + [a.replace("{d}", "") for _, a in VQA_PAIRS]
          + list("0123456789"))
tok = CharTokenizer(corpus)
print("词表大小:", tok.vocab_size, "| 词表:", "".join(tok.itos[3:]))

# ---------------- MNIST ----------------
tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.5,), (0.5,))])
# 视觉预训练用带数据增强的变换（轻微旋转/平移，提升泛化）
tf_aug = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
train_set = datasets.MNIST("./data", train=True, download=True, transform=tf)
train_set_aug = datasets.MNIST("./data", train=True, download=True, transform=tf_aug)
test_set = datasets.MNIST("./data", train=False, download=True, transform=tf)
train_sub = Subset(train_set_aug, range(60000))   # 完整训练集 + 增强
test_sub = Subset(test_set, range(1000))

def make_vlm_sample(img, digit, prompt, answer_tmpl):
    """input = prompt + answer(+EOS)；labels 只在 answer 段"""
    p_ids = tok.encode(prompt)
    a_ids = tok.encode(answer_tmpl.format(d=digit)) + [EOS_ID]
    return img, p_ids, a_ids

def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    max_len = max(len(b[1]) + len(b[2]) for b in batch)
    ids = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, (_, p, a) in enumerate(batch):
        ids[i, :len(p)+len(a)] = torch.tensor(p + a)
        labels[i, len(p):len(p)+len(a)] = torch.tensor(a)   # prompt 段保持 -100
    return imgs.to(DEVICE), ids.to(DEVICE), labels.to(DEVICE)

def gen_stage_samples(n, mode):
    """从 MNIST 子集随机抽样生成 n 条 (img, prompt, answer)"""
    out = []
    for _ in range(n):
        img, d = train_set[random.randrange(60000)]
        if mode == "caption":
            out.append(make_vlm_sample(img, d, CAPTION_PROMPT, CAPTION_TMPL))
        else:  # vqa
            q, a = random.choice(VQA_PAIRS)
            out.append(make_vlm_sample(img, d, q, a))
    return out

def run_epochs(model, samples, epochs, batch_size, optimizer, tag, params_note):
    losses = []
    for ep in range(epochs):
        random.shuffle(samples)
        tot, nb = 0.0, 0
        for i in range(0, len(samples), batch_size):
            imgs, ids, labels = collate(samples[i:i+batch_size])
            _, loss = model(imgs, ids, labels)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            tot += loss.item(); nb += 1
        avg = tot / nb; losses.append(avg)
        print(f"  [{tag}] epoch {ep+1}/{epochs} loss={avg:.4f} ({params_note})")
    return losses

log = {}

# ================= 阶段0a：ViT 视觉预训练（MNIST 分类） =================
print("\n===== 阶段0a：ViT 预训练（数字分类，教会'眼睛'） =====")
model = MiniVLM(tok.vocab_size).to(DEVICE)
opt = torch.optim.AdamW(model.vision.parameters(), lr=1e-3, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=6)
loader = DataLoader(train_sub, batch_size=256, shuffle=True)
t0 = time.time()
for ep in range(6):
    model.vision.train(); tot = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        loss = F.cross_entropy(model.vision.classify(imgs), labels)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    sched.step()
    print(f"  [ViT] epoch {ep+1}/6 loss={tot/len(loader):.4f}")
# 测试分类准确率
model.vision.eval(); correct = 0
with torch.no_grad():
    for imgs, labels in DataLoader(test_sub, batch_size=256):
        imgs = imgs.to(DEVICE)
        correct += (model.vision.classify(imgs).argmax(1).cpu() == labels).sum().item()
vit_acc = correct / len(test_sub)
log["stage0a_vit_acc"] = vit_acc
print(f"  ViT 分类准确率: {vit_acc*100:.2f}%  ({time.time()-t0:.0f}s)")

# ================= 阶段0b：LLM 文本预训练（教会'嘴巴'） =================
print("\n===== 阶段0b：LLM 预训练（纯文本语料，教会语言模板） =====")
lm_corpus = ([CAPTION_TMPL.format(d=d) for d in range(10)]
             + [a.format(d=d) for _, a in VQA_PAIRS for d in range(10)]) * 20
random.shuffle(lm_corpus)
opt = torch.optim.AdamW(model.llm.parameters(), lr=1e-3)
model.llm.train()
for ep in range(4):
    random.shuffle(lm_corpus); tot, nb = 0.0, 0
    for i in range(0, len(lm_corpus), 128):
        seqs = [tok.encode(s) + [EOS_ID] for s in lm_corpus[i:i+128]]
        ml = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), ml), PAD_ID, dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, :len(s)] = torch.tensor(s)
        ids = ids.to(DEVICE)
        logits = model.llm(input_ids=ids[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               ids[:, 1:].reshape(-1), ignore_index=PAD_ID)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item(); nb += 1
    print(f"  [LLM] epoch {ep+1}/4 loss={tot/nb:.4f}")

# ================= 阶段1：投影层对齐（冻结 ViT 与 LLM，只训投影层） =================
print("\n===== 阶段1：特征对齐（LLaVA 式，只训投影层） =====")
model.vision.eval(); model.llm.eval()          # 冻结模块同时关掉 dropout
for p in model.vision.parameters(): p.requires_grad = False
for p in model.llm.parameters(): p.requires_grad = False
model.projector.train()
opt = torch.optim.AdamW(model.projector.parameters(), lr=1e-3)
samples = gen_stage_samples(6000, "caption")
log["stage1"] = run_epochs(model, samples, 4, 128, opt, "align", "仅投影层 8,320 参数")

# ================= 阶段2：指令微调（解冻 LLM + 投影层，ViT 保持冻结） =================
print("\n===== 阶段2：视觉指令微调（投影层 + LLM，VQA） =====")
for p in model.llm.parameters(): p.requires_grad = True
model.llm.train()
opt = torch.optim.AdamW(
    list(model.projector.parameters()) + list(model.llm.parameters()),
    lr=5e-4, weight_decay=0.01)
# 数据混合：VQA : 描述 = 2 : 1，防止单任务微调导致的灾难性遗忘
samples = gen_stage_samples(4500, "vqa") + gen_stage_samples(2250, "caption")
log["stage2"] = run_epochs(model, samples, 4, 128, opt, "sft", "投影层+LLM 约11万参数")

# ================= 评估：测试集 VQA 准确率 + 生成样例 =================
print("\n===== 评估（1000 张未见测试图） =====")
model.eval()
q_test, a_test = VQA_PAIRS[0]
cap_prompt_ids = tok.encode(CAPTION_PROMPT)
vqa_correct = cap_correct = 0
with torch.no_grad():
    for j in range(len(test_sub)):
        img, d = test_sub[j]
        img = img.unsqueeze(0).to(DEVICE)
        if str(d) in tok.decode(model.generate(img, tok.encode(q_test))):
            vqa_correct += 1
        if str(d) in tok.decode(model.generate(img, cap_prompt_ids)):
            cap_correct += 1
log["vqa_acc@1000"] = vqa_correct / len(test_sub)
log["caption_acc@1000"] = cap_correct / len(test_sub)
print(f"  VQA 准确率: {log['vqa_acc@1000']*100:.2f}%")
print(f"  描述准确率: {log['caption_acc@1000']*100:.2f}%")

print("\n  生成样例（问题 → 模型输出 | 真实标签）")
for j in [0, 1, 2, 3, 4]:
    img, d = test_sub[j]
    for q, _ in [VQA_PAIRS[j % 3]]:
        out = model.generate(img.unsqueeze(0).to(DEVICE), tok.encode(q))
        print(f"    「{q}」→「{tok.decode(out)}」 | 真实: {d}")
    img, d = test_sub[j]
    out = model.generate(img.unsqueeze(0).to(DEVICE), tok.encode(CAPTION_PROMPT))
    print(f"    「{CAPTION_PROMPT}」→「{tok.decode(out)}」 | 真实: {d}")

torch.save({"model": model.state_dict(), "stoi": tok.stoi},
           f"{CKPT}/mini_vlm_final.pt")
with open(f"{CKPT}/train_log.json", "w") as f:
    json.dump(log, f, ensure_ascii=False, indent=2)
print("\n模型与日志已保存到", CKPT)
