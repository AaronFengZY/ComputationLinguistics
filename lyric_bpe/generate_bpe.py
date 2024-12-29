import torch
import numpy as np
import pickle
import sentencepiece as spm
from model import GPT, GPTConfig  # 你训练时用到的模型类

# =========== 需要你自行修改的路径 ===========
ckpt_path = "/home/v-zhifeng/HPE/nanoGPT/out-lyric-bpe/ckpt.pt"
meta_path = "/home/v-zhifeng/HPE/nanoGPT/data/lyric_bpe/meta.pkl"

# 该 Prompt 就是你的输入歌词
prompt_str = "你的柔情似水 ，几度让我爱得沉醉 ，毫无保留不知道后悔 ，你能不能体会真情可贵 ，没有余力伤悲 ，爱情像难收的覆水 ，长长来路走的太憔悴"

# =========== 选择设备 (GPU / CPU) ===========
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =========== 加载 meta.pkl ===========
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

vocab_size = meta.get("vocab_size", None)
sp_model_path = meta.get("sentencepiece_model", None)  # 在 prepare 阶段存储
print(f"Vocab size: {vocab_size}, spm model: {sp_model_path}")

# =========== 初始化分词器 (SentencePiece) ===========
sp = spm.SentencePieceProcessor()
if sp_model_path is None:
    raise ValueError("No 'sentencepiece_model' found in meta.pkl. Please check your meta or revise code.")
sp.load(sp_model_path)

# =========== 加载训练好的 GPT 模型 ===========
checkpoint = torch.load(ckpt_path, map_location=device)
model_args = checkpoint['model_args']
state_dict = checkpoint['model']

# 兼容可能存在的 "_orig_mod." 前缀
unwanted_prefix = "_orig_mod."
for k in list(state_dict.keys()):
    if k.startswith(unwanted_prefix):
        new_key = k[len(unwanted_prefix):]
        state_dict[new_key] = state_dict.pop(k)

# 用训练时的配置初始化模型
config = GPTConfig(**model_args)
model = GPT(config)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

print(f"Model loaded from {ckpt_path}.")

# =========== 定义一个简易的 sample 函数 (如无内置) ===========
@torch.no_grad()
def sample(model, idx, max_new_tokens=50, temperature=1.0, top_k=None):
    """
    从给定的 idx（prompt）继续生成 max_new_tokens 个 token。
    idx: (1, T) shape 的长整型张量
    """
    block_size = model.config.block_size
    for _ in range(max_new_tokens):
        # 如果序列已经超过block_size，则截断
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # 前向计算 logits
        logits, _ = model(idx_cond)
        # 取最后一个时间步
        logits = logits[:, -1, :] / temperature
        # Optional: top_k 策略
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')
        probs = torch.softmax(logits, dim=-1)
        # 从 probs 中采样一个 token
        next_token = torch.multinomial(probs, num_samples=1)
        # 连接到序列后面
        idx = torch.cat((idx, next_token), dim=1)
    return idx

# =========== 1) 用 SentencePiece 对 prompt_str 分词 ===========
prompt_ids = sp.encode(prompt_str, out_type=int)
prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)

# =========== 2) 使用 sample 函数生成若干新 tokens ===========
max_new_tokens = 100     # 续写长度，可以自由调
temperature = 0.8        # 温度，越高越随机，越低越保守
top_k = 40               # 只在概率前 top_k 中采样，可控制输出质量

gen_tokens = sample(
    model,
    idx=prompt_t,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_k=top_k
)

# =========== 3) 把生成后的 tokens decode 回中文文本 ===========
gen_tokens_list = gen_tokens[0].cpu().numpy().tolist()
gen_text = sp.decode(gen_tokens_list)

print("====================================")
print("        [模型续写结果示例]         ")
print("====================================")
print(gen_text)
print("====================================")

# 如果你只想看新增的那一段，可以：
# new_tokens = gen_tokens_list[len(prompt_ids):]
# new_text = sp.decode(new_tokens)
# print("新生成部分：", new_text)
