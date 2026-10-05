# VLM LeetCode 式练习

这套练习把根目录中的 MiniVLM 拆成 15 个可以独立完成的小问题。每一关只要求实现少量函数或类，并用单元测试给出明确的通过条件。

## 使用方式

1. 创建 Python 3.10+ 环境并安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

2. 从 `exercises/` 中按编号实现 TODO。
3. 一次只运行一关：

   ```bash
   pytest tests/test_01_tokenizer.py -q
   ```

4. 卡住时阅读同名的 `solutions/` 文件。验证参考实现：

   ```bash
   $env:VLM_IMPL="solutions"   # PowerShell
   pytest -q
   ```

   Linux/macOS 使用 `VLM_IMPL=solutions pytest -q`。

默认 `VLM_IMPL=exercises`，因此尚未填写的 TODO 导致测试失败是正常现象。

## 学习路线

| 关卡 | 主题 | 需要掌握的核心问题 |
|---|---|---|
| 01 | 字符分词器 | 文本怎样变成 token id，PAD/EOS 为什么必须分开 |
| 02 | 图像切块 | 卷积如何把图片变成 patch token |
| 03 | 正弦位置编码 | Transformer 如何感知 patch 的绝对位置 |
| 04 | 因果掩码 | 为什么答案 token 不能看到未来 |
| 05 | RoPE | 如何把相对位置信息旋转进 Q/K |
| 06 | 多头注意力 | Q/K/V、head 拆分和 causal attention |
| 07 | Transformer block | Pre-Norm、残差连接和 FFN |
| 08 | ViT 编码器 | CLS 分类与 patch 特征输出 |
| 09 | 视觉投影层 | 把视觉特征映射到语言嵌入空间 |
| 10 | 多模态拼接 | image tokens 如何作为 prefix 接入 LLM |
| 11 | 答案损失 | label shift、prompt mask、EOS 监督 |
| 12 | 分阶段冻结 | 对齐阶段与指令微调阶段训练哪些参数 |
| 13 | 自回归生成 | greedy decoding 与 EOS 停止 |
| 14 | Checkpoint | 模型、词表和配置如何一起保存恢复 |
| 15 | 训练步骤 | 清梯度、反向传播、梯度裁剪和参数更新的正确顺序 |

建议每关遵循相同节奏：先阅读测试，手算张量形状，填写 TODO，运行该关测试，最后再和参考答案比较。不要直接复制答案；先写出失败版本，测试反馈本身就是练习的一部分。

## 与原项目的对应关系

- 01 对应 `CharTokenizer`
- 02、03、08 对应 `ViTEncoder`
- 04～07 对应 `RoPEGPT`
- 09、10 对应 `Projector` 和 `MiniVLM.splice`
- 11 对应 `MiniVLM.forward` 与训练脚本的 `collate`
- 12 对应四阶段训练中的冻结/解冻逻辑
- 13 对应 `MiniVLM.generate`
- 14 补足原项目 checkpoint 元数据不完整的问题
- 15 对应训练脚本中每个 batch 的优化步骤

## 测试原则

测试主要检查：张量形状、数学不变量、梯度流向、损失对齐和保存恢复，而不是复现 README 中的最终准确率。因此绝大多数测试只需要 CPU，且不下载 MNIST。
