import os
import math
import torch
import numpy as np
import torch.nn as nn

from transformers import GPT2Config, GPT2LMHeadModel

# =========== 路径 ===============
ckpt_path = "/home/v-zhifeng/HPE/nanoGPT/out-lyric-gpt2-distil-chinese-bpe/ckpt_iter_5000.pt"
test_bin_path = "/home/v-zhifeng/HPE/nanoGPT/data/lyric_bpe/test.bin"

my_config = GPT2Config(
    vocab_size=12000,
    n_positions=1024,     # 非常重要，保持和训练时的一致
    n_embd=768,
    n_layer=12,
    n_head=12,
    n_ctx=1024,           # 同样保持一致
    tie_word_embeddings=False
)

block_size = 256  # 训练时常用的上下文长度
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =========== 定义 GPT2Wrapper (与train类似) ==========

class GPT2Wrapper(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, idx, targets=None):
        outputs = self.hf_model(input_ids=idx, labels=targets)
        return outputs.logits, outputs.loss

# =========== 构建空的 GPT2LMHeadModel ===========

hf_model = GPT2LMHeadModel(my_config)
hf_model.to(device)
model = GPT2Wrapper(hf_model).to(device)
model.eval()

# =========== 加载 checkpoint ===========

print(f"Loading checkpoint from {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=device)

if 'model_state' in checkpoint:
    state_dict = checkpoint['model_state']
else:
    state_dict = checkpoint

unwanted_prefix = '_orig_mod.'
for k in list(state_dict.keys()):
    if k.startswith(unwanted_prefix):
        new_k = k[len(unwanted_prefix):]
        state_dict[new_k] = state_dict.pop(k)

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)
print("Checkpoint loaded.")

# =========== 读入 test.bin ===========
test_data_np = np.fromfile(test_bin_path, dtype=np.uint32)
test_data = torch.from_numpy(test_data_np).long()
print(f"Test data has {test_data.size(0)} tokens total.")

# =========== 计算 PPL ===========
n = test_data.size(0) - 1
step = block_size
total_loss = 0.0
total_tokens = 0

with torch.no_grad():
    for i in range(0, n, step):
        if i + block_size >= n:
            break
        x = test_data[i : i+block_size].to(device)
        y = test_data[i+1 : i+1+block_size].to(device)

        # batch_size=1
        _, loss = model(x.unsqueeze(0), y.unsqueeze(0))
        total_loss += loss.item() * block_size
        total_tokens += block_size

avg_loss = total_loss / total_tokens
ppl = math.exp(avg_loss)
print(f"Test set average loss: {avg_loss:.4f}")
print(f"Test set perplexity :  {ppl:.4f}")
