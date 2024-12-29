import os
import torch
import numpy as np
import math
from torch.nn import Module
from transformers import (
    GPT2LMHeadModel,
    BertTokenizer  # uer/gpt2-distil-chinese 用的BertTokenizer
)

# =========== 需要你自行修改的路径 ===========
ckpt_path = "/home/v-zhifeng/HPE/nanoGPT/out-lyric-gpt2-distil-chinese/ckpt.pt"

# 该 Prompt 就是你的输入歌词
prompt_str = "你的柔情似水 ，几度让我爱得沉醉 ，毫无保留不知道后悔 ，你能不能体会真情可贵 ，没有余力伤悲 ，爱情像难收的覆水 ，长长来路走的太憔悴"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = 'bfloat16'  # 若GPU不支持可设 'float16' 或 'float32'
print(f"Using device: {device}, dtype={dtype}")

# ------------------ 1. 加载 ckpt ------------------
checkpoint = torch.load(ckpt_path, map_location=device)
model_state = checkpoint["model_state"]

# ------------------ 2. 构造与训练时一致的模型结构 ------------------
class GPT2Wrapper(Module):
    def __init__(self, pretrained_name="uer/gpt2-distil-chinese-cluecorpussmall"):
        super().__init__()
        self.hf_model = GPT2LMHeadModel.from_pretrained(pretrained_name)

    def forward(self, idx, targets=None):
        outputs = self.hf_model(input_ids=idx, labels=targets)
        return outputs.logits, outputs.loss

model = GPT2Wrapper("uer/gpt2-distil-chinese-cluecorpussmall")
model.load_state_dict(model_state, strict=False)
model.to(device)
if dtype == 'bfloat16':
    model.half()
model.eval()
print("Model loaded from", ckpt_path)

# ------------------ 3. 初始化 tokenizer ------------------
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")

# ------------------ 4. 定义推理函数 (逐token生成) ------------------
@torch.no_grad()
def sample(model, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
    """
    input_ids: (1, T)  已经是tensor形式
    """
    block_size = 256  # 必须与微调时保持一致
    for _ in range(max_new_tokens):
        idx_cond = input_ids if input_ids.size(1) <= block_size else input_ids[:, -block_size:]
        outputs = model(idx_cond)
        logits = outputs[0]  # (batch_size=1, seq_len, vocab_size)
        logits = logits[:, -1, :] / temperature  # 取最后一个token的logits
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # [1, 1]
        input_ids = torch.cat((input_ids, next_id), dim=1)
    return input_ids

# ------------------ 5. Encode prompt, generate, decode ------------------
# encode
prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

# generate
max_new_tokens = 100
temperature = 0.8
top_k = 40

gen_ids = sample(model, input_ids, max_new_tokens, temperature, top_k)
gen_ids_list = gen_ids[0].cpu().tolist()

# decode
gen_text = tokenizer.decode(gen_ids_list, clean_up_tokenization_spaces=True)

print("====================================")
print("        [模型续写结果示例]         ")
print("====================================")
print(gen_text)
print("====================================")

# 如果你只想看新生成的部分
# new_tokens = gen_ids_list[len(prompt_ids):]
# new_text = tokenizer.decode(new_tokens, clean_up_tokenization_spaces=True)
# print("新生成部分：", new_text)
