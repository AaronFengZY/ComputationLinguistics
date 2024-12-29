import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2LMHeadModel

# --------------------- 超参数 / 配置 ---------------------
out_dir = 'out-lyric-gpt2-distil-chinese'
eval_interval = 1000       # 每隔多少 iter 做一次eval
eval_iters = 200          # eval时一共跑多少个batch估算loss
eval_only = False         # 是否仅评估
log_interval = 10         # 训练期间的信息打印间隔
always_save_checkpoint = True

init_from = 'huggingface'                        # 关键点：从 huggingface 预训练模型加载
pretrained_name = 'uer/gpt2-distil-chinese-cluecorpussmall'

# 是否启用 wandb 记录
wandb_log = True
wandb_project = 'lyric-gpt2-distil-chinese'
wandb_run_name = 'mini-gpt2-distil-chinese'

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu'
# 尝试使用 bfloat16，如不支持则自动使用 float16
dtype = 'bfloat16' if (device_type == 'cuda' and torch.cuda.is_bf16_supported()) else 'float16'
compile_model = False  # 是否使用 PyTorch 2.0 的 compile 加速


# 数据路径
data_dir = 'data/lyric_gpt2_distil_chinese'

# 训练超参(参照 mini 配置)
batch_size = 64
block_size = 256
max_iters = 5000
lr_decay_iters = 5000   # 通常设置为与 max_iters 相同
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

learning_rate = 1e-3
weight_decay = 0.0    # 你可以适度加大，如 0.01，看任务需要

# --------------------- DDP 多卡设置 ---------------------
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ptdtype = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16
}
use_dtype = ptdtype[dtype]

print(f"Using device={device}, dtype={dtype}, master_process={master_process}")

# (可选) wandb
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name)

# --------------------- 数据加载函数 ---------------------
def get_data(split):
    # 例如: "train.bin", "val.bin"
    bin_name = 'train.bin' if split == 'train' else 'val.bin'
    path = os.path.join(data_dir, bin_name)
    data = np.fromfile(path, dtype=np.uint32)  # 与 tokenizer 分词一致
    return data

train_data = get_data('train')
val_data = get_data('val')

def get_batch(split):
    """
    随机采样一个 batch:
    X, Y: (batch_size, block_size)
    """
    data = train_data if split == 'train' else val_data
    ix = np.random.randint(0, len(data) - block_size - 1, (batch_size,))
    # 逐条截取
    Xb = [data[i : i + block_size] for i in ix]
    Yb = [data[i + 1 : i + 1 + block_size] for i in ix]
    Xb = torch.tensor(Xb, dtype=torch.long, device=device)
    Yb = torch.tensor(Yb, dtype=torch.long, device=device)
    return Xb, Yb

# --------------------- 学习率衰减函数 ---------------------
def get_lr(iter):
    # linear warmup
    if iter < warmup_iters:
        return learning_rate * (iter + 1) / warmup_iters
    # then cosine decay
    if iter > lr_decay_iters:
        return min_lr
    ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    # cosine decay from learning_rate to min_lr
    decay = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + (learning_rate - min_lr) * decay

# --------------------- 模型初始化部分 ---------------------
if init_from == 'huggingface':
    print(f"Loading pretrained model: {pretrained_name}")
    hf_model = GPT2LMHeadModel.from_pretrained(pretrained_name)
    hf_model.to(device)

    # 构建一个包装器，以便与 nanoGPT 相似的调用
    class GPT2Wrapper(nn.Module):
        def __init__(self, hf_model):
            super().__init__()
            self.hf_model = hf_model

        def forward(self, idx, targets=None):
            # idx: (batch_size, block_size)
            out = self.hf_model(input_ids=idx, labels=targets)
            # out.loss 是标量 cross entropy； out.logits形状 (batch_size, seq_len, vocab_size)
            return out.logits, out.loss

    model = GPT2Wrapper(hf_model)
else:
    raise ValueError("Only huggingface init is supported in this example.")

# optional: compile
if compile_model:
    print("Compiling model...")
    model = torch.compile(model)

# 如果需要 DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# --------------------- 优化器 ---------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, beta2), weight_decay=weight_decay)

# --------------------- 评估函数 ---------------------
@torch.no_grad()
def estimate_loss():
    """ 在 train/val 上估算平均loss """
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses.append(loss.item())
        out[split] = float(np.mean(losses))
    model.train()
    return out

# --------------------- 训练循环 ---------------------
best_val_loss = float('inf')
iter_num = 0
for iter_num in range(max_iters + 1):

    print("Iteration", iter_num)
    # 每个 iter 可能先 eval
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        train_loss, val_loss = losses['train'], losses['val']
        print(f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]['lr'],
            })

        # 保存 checkpoint
        if iter_num == max_iters:
            best_val_loss = val_loss
            ckpt_path = os.path.join(out_dir, "ckpt.pt")
            raw_model = model.module if ddp else model
            checkpoint = {
                'iter_num': iter_num,
                'model_state': raw_model.state_dict(),
            }
            print(f"Saving checkpoint to {ckpt_path}")
            torch.save(checkpoint, ckpt_path)

    if eval_only:
        break

    # 获取一个 batch
    X, Y = get_batch('train')

    # 设置学习率
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 前向 + 反向
    logits, loss = model(X, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 收尾
if ddp:
    destroy_process_group()
print("Training finished.")
