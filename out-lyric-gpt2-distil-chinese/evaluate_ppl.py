import os
import math
import torch
import numpy as np
from torch.nn import Module
from transformers import GPT2LMHeadModel

# =========== 需要你自行修改的路径 ===========
ckpt_path = "/home/v-zhifeng/HPE/nanoGPT/out-lyric-gpt2-distil-chinese/ckpt.pt"
test_bin_path = "/home/v-zhifeng/HPE/nanoGPT/data/lyric_gpt2_distil_chinese/test.bin"

# 训练时的超参或设定
block_size = 256   # 和你在微调时的 block_size 保持一致
dtype = 'bfloat16' # 如果你训练时用的 bf16，且GPU支持，就保持，否则可设 'float16'
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}, dtype={dtype}")

# ------------------ 1. 加载 ckpt ------------------
checkpoint = torch.load(ckpt_path, map_location=device)
print(f"Loaded checkpoint from {ckpt_path} with keys: {list(checkpoint.keys())}")

# 从 checkpoint 中提取模型权重
model_state = checkpoint["model_state"]

# ------------------ 2. 构造与训练时一致的模型结构 ------------------
# 训练时是 huggingface GPT-2 + 一个 Wrapper

class GPT2Wrapper(Module):
    def __init__(self, pretrained_name="uer/gpt2-distil-chinese-cluecorpussmall"):
        super().__init__()
        self.hf_model = GPT2LMHeadModel.from_pretrained(pretrained_name)

    def forward(self, idx, targets=None):
        # idx: [batch_size, seq_len]
        outputs = self.hf_model(input_ids=idx, labels=targets)
        # outputs.loss: scalar, outputs.logits: [batch_size, seq_len, vocab_size]
        return outputs.logits, outputs.loss

# 初始化
model = GPT2Wrapper("uer/gpt2-distil-chinese-cluecorpussmall")
model.load_state_dict(model_state, strict=False)  # 如果提示 missing keys, 确认一下
model.to(device)
if dtype == 'bfloat16':
    model.half()  # bfloat16 / half() 通常是 fp16，但实际 hf_model 也支持半精度
model.eval()

print("Model loaded and set to eval mode.")

# ------------------ 3. 读取 test.bin 并转为 Torch Tensor ------------------
test_data_np = np.fromfile(test_bin_path, dtype=np.uint32)  # 训练前prepare时用np.uint32
test_data = torch.from_numpy(test_data_np).long()
print(f"Test data has {test_data.size(0)} tokens in total.")

# ------------------ 4. 计算 PPL ------------------
total_loss = 0.0
total_tokens = 0

n = test_data.size(0) - 1  # 前 n 个 token 作为输入，预测其后一个
step = block_size

with torch.no_grad():
    for i in range(0, n, step):
        if i + block_size >= n:
            break
        x = test_data[i : i + block_size].to(device)
        y = test_data[i + 1 : i + block_size + 1].to(device)
        # batch_size=1
        # forward
        _, loss = model(x.unsqueeze(0), y.unsqueeze(0))
        # loss.item() 是该 batch 的平均 cross-entropy
        # 乘以 block_size 再累加，以得到整体 sum of loss
        total_loss += loss.item() * block_size
        total_tokens += block_size

avg_loss = total_loss / total_tokens
ppl = math.exp(avg_loss)

print(f"Test set average loss: {avg_loss:.4f}")
print(f"Test set perplexity :  {ppl:.4f}")
