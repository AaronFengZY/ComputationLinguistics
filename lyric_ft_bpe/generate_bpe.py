import os
import torch
import numpy as np
import sentencepiece as spm
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

# =========== 1. 需要你修改的路径/配置 ===========
ckpt_path = "/home/v-zhifeng/HPE/nanoGPT/out-lyric-gpt2-distil-chinese-bpe/ckpt_iter_5000.pt"
sp_model_path = "/home/v-zhifeng/HPE/nanoGPT/lyrics.model"

my_config = GPT2Config(
    vocab_size=12000,       # 与微调时一致
    n_positions=1024,       # GPT2默认1024, 与训练时一致
    n_embd=768,
    n_layer=12,
    n_head=12,
    n_ctx=1024,
    tie_word_embeddings=False
)

# 设定采样参数
max_new_tokens = 100
temperature = 0.8
top_k = 40

prompt_str = "你的柔情似水 ，几度让我爱得沉醉 ，毫无保留不知道后悔 ，你能不能体会真情可贵 ，没有余力伤悲 ，爱情像难收的覆水 ，长长来路走的太憔悴"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =========== 2. GPT2Wrapper 类（与训练/推理保持一致） ===========
class GPT2Wrapper(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, idx, targets=None):
        # idx: (batch_size, seq_len)
        outputs = self.hf_model(input_ids=idx, labels=targets)
        # outputs.loss: [scalar cross_entropy], outputs.logits: [batch, seq_len, vocab_size]
        return outputs.logits, outputs.loss

# =========== 3. 构造空白 GPT2LMHeadModel 并封装成 GPT2Wrapper ===========
hf_model = GPT2LMHeadModel(my_config)
hf_model.to(device)

model = GPT2Wrapper(hf_model)
model.to(device)
model.eval()

# =========== 4. 加载微调后的 checkpoint ===========
print(f"Loading checkpoint from {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=device)

# 若 checkpoint 是 {'model_state': state_dict, ...}, 则:
if 'model_state' in checkpoint:
    state_dict = checkpoint['model_state']
else:
    # 如果直接是 state_dict
    state_dict = checkpoint

# 若存在 '_orig_mod.' 前缀，去除
unwanted_prefix = "_orig_mod."
for k in list(state_dict.keys()):
    if k.startswith(unwanted_prefix):
        new_key = k[len(unwanted_prefix):]
        state_dict[new_key] = state_dict.pop(k)

# 加载权重
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)
print("Checkpoint loaded successfully.")

# =========== 5. 初始化 SentencePiece 分词器 ===============
sp = spm.SentencePieceProcessor()
sp.load(sp_model_path)
print(f"SentencePiece model loaded from {sp_model_path}.")

# =========== 6. 定义一个 sample 函数，用于文本续写 ===========
@torch.no_grad()
def sample(model, idx, max_new_tokens=50, temperature=1.0, top_k=None):
    block_size = model.hf_model.config.n_positions  # or 1024
    for _ in range(max_new_tokens):
        # 如果序列已达或超过block_size，则截断
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

        logits, _ = model(idx_cond)
        # 取最后一个时间步 logits
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [1,1]
        idx = torch.cat([idx, next_token], dim=1)
    return idx

# =========== 7. 将 prompt_str 分词，调用 sample 函数 ===========
prompt_ids = sp.encode(prompt_str, out_type=int)
prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)

gen_tokens = sample(model, idx=prompt_t, max_new_tokens=max_new_tokens,
                    temperature=temperature, top_k=top_k)

gen_tokens_list = gen_tokens[0].cpu().numpy().tolist()
# 解码回中文
gen_text = sp.decode(gen_tokens_list)

print("====================================")
print("         [模型续写结果示例]         ")
print("====================================")
print(gen_text)
print("====================================")

