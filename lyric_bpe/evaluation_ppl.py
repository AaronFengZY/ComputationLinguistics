import os
import math
import pickle
import torch
import numpy as np

# 如果你的模型类定义在 model.py 或其他文件，请确保可正确导入
from model import GPT, GPTConfig  # 你训练时用到的模型类

# =========== 需要修改的路径 ===========
ckpt_path = "/home/v-zhifeng/HPE/nanoGPT/out-lyric-bpe-lr5e-2-iters10000/ckpt.pt"
test_bin_path = "/home/v-zhifeng/HPE/nanoGPT/data/lyric_bpe/test.bin"
meta_path = "/home/v-zhifeng/HPE/nanoGPT/data/lyric_bpe/meta.pkl"

# =========== 选择推理设备 ===========
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --------------------------------------------------------
# 1. 载入 meta.pkl，获取 vocab_size 等信息
#    （如果你有自定义 merges/vocab 也可在这里读取）
# --------------------------------------------------------
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

vocab_size = meta.get('vocab_size', None)
print(f"Vocabulary size (from meta): {vocab_size}")

# --------------------------------------------------------
# 2. 加载训练好的 checkpoint
# --------------------------------------------------------
checkpoint = torch.load(ckpt_path, map_location=device)
model_args = checkpoint['model_args']
state_dict = checkpoint['model']

# 如果训练脚本中可能有 "_orig_mod." 前缀，需要移除
unwanted_prefix = "_orig_mod."
for k in list(state_dict.keys()):
    if k.startswith(unwanted_prefix):
        new_key = k[len(unwanted_prefix):]
        state_dict[new_key] = state_dict.pop(k)

# 用与训练一致的 GPTConfig 来初始化模型
config = GPTConfig(**model_args)
model = GPT(config)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

print("Model loaded from", ckpt_path)

# --------------------------------------------------------
# 3. 读取 test.bin 并转换为 Torch Tensor
#    注意 dtype=np.uint32 (与 prepare 时相匹配)
# --------------------------------------------------------
test_data_np = np.fromfile(test_bin_path, dtype=np.uint16)
test_data = torch.from_numpy(test_data_np).long()
print(f"Test data has {test_data.size(0)} tokens in total.")

# --------------------------------------------------------
# 4. 计算 Perplexity (PPL)
# --------------------------------------------------------
# 常见做法：基于 block_size 逐段测试，计算平均 cross-entropy
block_size = model.config.block_size  # 通常在训练时就设定，如 1024
total_loss = 0.0
total_tokens = 0

n = test_data.size(0) - 1  # 要预测下一个 token，所以只有前 n 个可作为输入
step = block_size  # 每次前向计算 block_size 长度

with torch.no_grad():
    for i in range(0, n, step):
        # 防止最后一段不足 block_size，可以截断或跳过
        if i + block_size >= n:
            break

        x = test_data[i : i + block_size].to(device)
        y = test_data[i + 1 : i + block_size + 1].to(device)

        # 将 batch_size 设为 1，即 [1, block_size]
        logits, loss = model(x.unsqueeze(0), y.unsqueeze(0))
        # loss.item() 是该 block 的平均交叉熵 (平均到 seq_len)
        # 为了更精确，将其乘以 block_size 再累加
        total_loss += loss.item() * block_size
        total_tokens += block_size

avg_loss = total_loss / total_tokens
ppl = math.exp(avg_loss)

print(f"Test set average loss: {avg_loss:.4f}")
print(f"Test set perplexity :  {ppl:.4f}")
