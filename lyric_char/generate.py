import os
import torch
import pickle
import numpy as np

# 这里假设你的模型定义在 model.py 中
from model import GPTConfig, GPT

# 1. 指定路径
ckpt_path = "/home/v-zhifeng/HPE/nanoGPT/out-lyric-char/ckpt.pt"
meta_path = "/home/v-zhifeng/HPE/nanoGPT/data/lyric_char/meta.pkl"

# 2. 选择推理设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 3. 加载 meta 数据
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi = meta["stoi"]
itos = meta["itos"]
vocab_size = meta["vocab_size"]

# 4. 加载模型
checkpoint = torch.load(ckpt_path, map_location=device)
model_args = checkpoint['model_args']
state_dict = checkpoint['model']

# 有时会遇到 state_dict 带 "_orig_mod." 前缀，可做一下兼容处理
unwanted_prefix = "_orig_mod."
for k in list(state_dict.keys()):
    if k.startswith(unwanted_prefix):
        new_key = k[len(unwanted_prefix):]
        state_dict[new_key] = state_dict.pop(k)

# 用相同 config 初始化模型并加载权重
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

print(f"Model loaded from {ckpt_path}. Ready for generation on {device}.")

# =========== 定义一个简易的 sample 函数，如果 model.py 没有内置的话 ===========
@torch.no_grad()
def sample(model, idx, max_new_tokens=50, temperature=1.0, top_k=None):
    """
    从给定的 idx（prompt）继续生成 max_new_tokens 个 token。
    idx: (1, T) shape的长整型张量，表示当前上下文
    """
    block_size = model.config.block_size
    for _ in range(max_new_tokens):
        # 如果序列已经达到或超过 block_size，就截断最老的 token
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # 前向计算 logits
        logits, _ = model(idx_cond)
        # 取最后一个时间步
        logits = logits[:, -1, :] / temperature
        # Optional: top_k 策略
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = torch.softmax(logits, dim=-1)
        # 从 probs 中采样
        next_token = torch.multinomial(probs, num_samples=1)
        # 连接到序列后面
        idx = torch.cat((idx, next_token), dim=1)
    return idx

# =========== 输入测试歌词（Prompt） ===========
prompt_str = "你的柔情似水 ，几度让我爱得沉醉 ，毫无保留不知道后悔 ，你能不能体会真情可贵 ，没有余力伤悲 ，爱情像难收的覆水 ，长长来路走的太憔悴"

# 将 prompt 编码为 token ID
# 注意：如果某些字符不在 stoi 中，可能会 KeyError，需要做相应处理(unk)或扩充词表
encoded_prompt = []
for ch in prompt_str:
    if ch in stoi:
        encoded_prompt.append(stoi[ch])
    else:
        # 如果需要处理未知字符，这里可以跳过或映射到 <UNK>
        # 这里简单跳过
        pass

# 转成 PyTorch 张量 (batch_size=1, seq_len=len_prompt)
prompt_t = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)

# =========== 调用 sample 函数生成文本 ===========
# 这里我们设置：
#   - max_new_tokens: 生成多少个 token
#   - temperature: 控制模型输出的“发散度”
#   - top_k: 只在概率前 k 的 token 中选，避免结果太随意
gen_tokens = sample(
    model, 
    idx=prompt_t, 
    max_new_tokens=100,  # 你可以改长一点或短一点
    temperature=0.8,
    top_k=40
)

# 取后续新生成的部分
gen_tokens = gen_tokens[0].cpu().numpy().tolist()
# 将所有 token decode 回中文字符串
gen_text = ''.join(itos[t] for t in gen_tokens)

print("====================================")
print("            [模型续写示例]           ")
print("====================================")
print(gen_text)
print("====================================")

# 你也可以只打印 prompt 之后的部分：
# new_tokens = gen_tokens[len(encoded_prompt):]  # 新生成段
# new_text = ''.join(itos[t] for t in new_tokens)
# print(new_text)
