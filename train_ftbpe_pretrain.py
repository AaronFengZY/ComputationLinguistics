import os
import math
import numpy as np
import torch
import torch.nn as nn
import sentencepiece as spm
from transformers import GPT2LMHeadModel, BertTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# ----------------- 配置 -----------------
out_dir = 'out-lyric-gpt2-debug'
eval_interval = 500
eval_iters = 50
log_interval = 10
always_save_checkpoint = True

init_from = 'huggingface'
pretrained_name = 'uer/gpt2-distil-chinese-cluecorpussmall'

sp_model_path = "/home/v-zhifeng/HPE/nanoGPT/lyrics.model"
data_dir = "data/lyric_bpe"

# 训练超参 (更保守)
batch_size = 32
block_size = 256
max_iters = 5000
learning_rate = 1e-5
beta2 = 0.99
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 1e-6

device = 'cuda:2'
# 关键: 用 float32 避免半精度导致溢出
dtype = 'float32' 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# wandb
wandb_log = True
wandb_project = 'debug-gpt2'
wandb_run_name = 'ft-bpe-no-nan'

# DDP
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
print(f"Using device={device}, dtype={dtype}, master_process={master_process}")

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name)

# ============ 1. 加载 GPT2 模型, 禁用 tying ============
if init_from != 'huggingface':
    raise ValueError("Only huggingface init is supported.")

hf_model = GPT2LMHeadModel.from_pretrained(pretrained_name)
hf_model.config.tie_word_embeddings = False  # 禁用 tying
hf_model.to(device)

old_vocab_size = hf_model.config.vocab_size
hidden_size = hf_model.config.n_embd
print(f"Loaded {pretrained_name}, old_vocab_size={old_vocab_size}, hidden_size={hidden_size}")

old_tokenizer = BertTokenizer.from_pretrained(pretrained_name)
old_embedding = hf_model.transformer.wte.weight.data  # [old_vocab_size, hidden_size]
old_lm_head_weight = hf_model.lm_head.weight.data
old_lm_head_bias = hf_model.lm_head.bias
if old_lm_head_bias is not None:
    old_lm_head_bias = old_lm_head_bias.data.clone()

# ============ 2. 新的SentecePiece词表 & 函数 ============
sp = spm.SentencePieceProcessor()
sp.load(sp_model_path)
new_vocab_size = sp.vocab_size()
print(f"new_vocab_size={new_vocab_size}")

def average_init_for_token_embed(piece_str):
    old_ids = old_tokenizer.encode(piece_str, add_special_tokens=False)
    valid_ids = [tid for tid in old_ids if tid < old_vocab_size]
    if len(valid_ids)==0:
        return None
    vec = torch.zeros(hidden_size, dtype=old_embedding.dtype, device=old_embedding.device)
    for tid in valid_ids:
        vec += old_embedding[tid]
    vec /= len(valid_ids)
    return vec

def average_init_for_token_head(piece_str):
    old_ids = old_tokenizer.encode(piece_str, add_special_tokens=False)
    valid_ids = [tid for tid in old_ids if tid<old_vocab_size]
    if len(valid_ids)==0:
        return None
    vec = torch.zeros(hidden_size, dtype=old_lm_head_weight.dtype, device=old_lm_head_weight.device)
    for tid in valid_ids:
        vec += old_lm_head_weight[tid]
    vec /= len(valid_ids)
    return vec

# ============ 3. 构建新embedding & lm_head, 分开做平均 =============
with torch.no_grad():
    new_embed = nn.Embedding(new_vocab_size, hidden_size, device=device, dtype=torch.float32)
    new_lm_head = nn.Linear(hidden_size, new_vocab_size, bias=True, device=device, dtype=torch.float32)

    nn.init.normal_(new_embed.weight, mean=0.0, std=0.01)  # std小一点
    nn.init.normal_(new_lm_head.weight, mean=0.0, std=0.01)
    nn.init.zeros_(new_lm_head.bias)

    c_embed, c_head = 0, 0
    for new_id in range(new_vocab_size):
        piece_str = sp.id_to_piece(new_id)

        v_embed = average_init_for_token_embed(piece_str)
        if v_embed is not None:
            new_embed.weight[new_id] = v_embed
            c_embed+=1

        v_head = average_init_for_token_head(piece_str)
        if v_head is not None:
            new_lm_head.weight[new_id] = v_head
            new_lm_head.bias[new_id] = 0.0
            c_head+=1

    print(f"[Init] embed matched={c_embed}, head matched={c_head}")
    hf_model.transformer.wte = new_embed
    hf_model.lm_head = new_lm_head
    hf_model.config.vocab_size = new_vocab_size

# ============ 4. Wrapper ============
class GPT2Wrapper(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, idx, targets=None):
        out = self.hf_model(input_ids=idx, labels=targets)
        return out.logits, out.loss

model = GPT2Wrapper(hf_model).to(device)

# 强制float32:
for p in model.parameters():
    p.data = p.data.float()

# ============ 5. 读入 .bin 数据 ============
def get_data(split):
    path = os.path.join(data_dir, 'train.bin' if split=='train' else 'val.bin')
    return np.fromfile(path, dtype=np.uint32)

train_data = get_data('train')
val_data = get_data('val')

def get_batch(split):
    data = train_data if split=='train' else val_data
    ix = np.random.randint(0, len(data)-block_size-1, size=batch_size)
    x_l = [data[i:i+block_size] for i in ix]
    y_l = [data[i+1:i+1+block_size] for i in ix]
    x_m = np.stack(x_l, axis=0)
    y_m = np.stack(y_l, axis=0)
    Xb = torch.tensor(x_m, dtype=torch.long, device=device)
    Yb = torch.tensor(y_m, dtype=torch.long, device=device)
    return Xb, Yb

# ============ 6. 优化器 & grad clip ============
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9,beta2))

def get_lr(step):
    if step < warmup_iters:
        return learning_rate*(step+1)/warmup_iters
    if step > lr_decay_iters:
        return min_lr
    ratio=(step-warmup_iters)/(lr_decay_iters-warmup_iters)
    decay=0.5*(1+math.cos(math.pi*ratio))
    return min_lr+(learning_rate-min_lr)*decay

# ============ 7. 定义一个检查NaN的函数 ============
def check_nan_in_params(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"[Warn] Found NaN in param {name}")
            return True
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"[Warn] Found NaN in grad of {name}")
            return True
    return False

# ============ 8. 评估函数 =============
@torch.no_grad()
def estimate_loss():
    model.eval()
    out={}
    for split in ['train','val']:
        losses=[]
        for _ in range(eval_iters):
            X, Y= get_batch(split)
            _, loss= model(X,Y)
            losses.append(loss.item())
        out[split]=float(np.mean(losses))
    model.train()
    return out

if ddp:
    model=DDP(model, device_ids=[int(device.split(':')[-1])])

# ============ 9. 训练循环 =============
best_val_loss=float('inf')
for iter_num in range(max_iters+1):
    # batch
    X,Y= get_batch('train')
    logits, loss=model(X,Y)
    optimizer.zero_grad()
    loss.backward()

    # grad clip
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

    # 检查是否出现NaN
    if check_nan_in_params(model):
        print(f"iter {iter_num}: found NaN, skip this step, set grad=0")
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
    else:
        optimizer.step()

    # lr
    lr= get_lr(iter_num)
    for pg in optimizer.param_groups:
        pg['lr']= lr

    # 打印 train loss
    if iter_num % log_interval==0:
        print(f"iter {iter_num}: train loss {loss.item():.4f}")
        if wandb_log and master_process:
            import wandb
            wandb.log({"iter":iter_num, "train_loss_iter": loss.item()})

    # eval
    if (iter_num%eval_interval==0 and iter_num>0) or iter_num==max_iters:
        losses=estimate_loss()
        trl, val= losses['train'], losses['val']
        print(f"step {iter_num}: train loss {trl:.4f}, val loss {val:.4f}")
        if wandb_log and master_process:
            wandb.log({"iter":iter_num, "train_loss_eval": trl, "val_loss_eval": val, "lr":lr})

        ckpt_path=os.path.join(out_dir,f"ckpt_iter_{iter_num}.pt")
        print(f"Saving ckpt to {ckpt_path}")
        raw_model=model.module if ddp else model
        torch.save({
            'iter_num': iter_num,
            'model_state': raw_model.state_dict(),
        }, ckpt_path)

        if val<best_val_loss:
            best_val_loss=val

print("Training finished.")
if ddp:
    destroy_process_group()
