import os
import math
import pickle
import torch
import numpy as np

from model import GPT, GPTConfig  # 你训练时使用的模型类

# ==== 需要修改的路径 ====
ckpt_path = "/home/v-zhifeng/HPE/nanoGPT/out-lyric-char/ckpt.pt"
test_bin_path = "/home/v-zhifeng/HPE/nanoGPT/data/lyric_char/test.bin"
meta_path = "/home/v-zhifeng/HPE/nanoGPT/data/lyric_char/meta.pkl"

# 选择设备：如果有GPU则使用GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --------------------------------------------------------
# 1. 载入 meta.pkl，获取词表大小、stoi/itos等信息
# --------------------------------------------------------
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']
print(f"Vocabulary size = {vocab_size}")

# --------------------------------------------------------
# 2. 加载训练好的 checkpoint
#    （包括模型权重、模型配置、优化器状态等）
# --------------------------------------------------------
checkpoint = torch.load(ckpt_path, map_location=device)
model_args = checkpoint['model_args']  # 在训练脚本里保存的 model 参数
state_dict = checkpoint['model']       # 模型权重

# 构建 GPTConfig，并用它初始化模型
config = GPTConfig(**model_args)
model = GPT(config)

# 有时会遇到 state_dict 里带有 "_orig_mod." 前缀，这里可以做个兼容处理:
unwanted_prefix = '_orig_mod.'
for k in list(state_dict.keys()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.to(device)
model.eval()  # 推理模式

print("Model loaded from", ckpt_path)

# --------------------------------------------------------
# 3. 读入 test.bin
#    说明：
#    - test.bin 里是 uint16 类型，每个元素是一条 token id
# --------------------------------------------------------
test_data_np = np.fromfile(test_bin_path, dtype=np.uint16)
test_data = torch.from_numpy(test_data_np).long()
print(f"Test data has {test_data.size(0)} tokens.")

# --------------------------------------------------------
# 4. 计算 Perplexity
#    思路：
#    - 按 block_size 切分 test_data
#    - 对每个切块都计算 cross entropy
#    - 得到平均 loss，再取 exp 即可得到 PPL
# --------------------------------------------------------
block_size = model.config.block_size

total_loss = 0.0
total_tokens = 0

# 因为要预测下一个 token，只有前 n-1 个 token 能作为输入
n = test_data.size(0) - 1

step = block_size

with torch.no_grad():
    for i in range(0, n, step):
        # 最后一段如果不足一个 block_size 就跳过，或也可部分计算
        if i + block_size >= n:
            break

        # inputs:   test_data[i : i+block_size]
        # targets:  test_data[i+1 : i+block_size+1]
        x = test_data[i : i + block_size].to(device)
        y = test_data[i + 1 : i + block_size + 1].to(device)

        # 前向计算，获取loss
        # GPT的 forward 通常返回 (logits, loss)
        logits, loss = model(x.unsqueeze(0), y.unsqueeze(0))  # [batch=1, seq=block_size]
        
        # loss.item() 是此段的平均 cross-entropy
        # 为了加权正确，我们乘以 block_size 再累加
        total_loss += loss.item() * block_size
        total_tokens += block_size

# 计算平均 loss
avg_loss = total_loss / total_tokens
ppl = math.exp(avg_loss)

print(f"Test set average loss: {avg_loss:.4f}")
print(f"Test set perplexity:   {ppl:.4f}")
