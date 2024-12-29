"""
sample.py: Generate text from a fine-tuned GPT2-like model
               supporting top-k and typical-p sampling
Usage:
  python sample.py \
    --ckpt "/home/v-zhifeng/HPE/nanoGPT/out-lyric-gpt2-distil-chinese/ckpt.pt" \
    --prompt "你的柔情似水..." \
    --num_samples 5 \
    --max_new_tokens 100 \
    --temperature 0.8 \
    --top_k 40 \
    --typical_p 0.9 \
    --method typical  # or "topk" to switch method
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, BertTokenizer

# ------------------------------------------------------------------
# 1. Parse command-line arguments
# ------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True,
                    help="Path to your model checkpoint, e.g. ckpt.pt")
parser.add_argument("--prompt", type=str, default="",
                    help="Your lyric prompt (UTF-8) to feed as input.")
parser.add_argument("--num_samples", type=int, default=5,
                    help="Number of samples to generate")
parser.add_argument("--max_new_tokens", type=int, default=50,
                    help="Max tokens to generate")
parser.add_argument("--temperature", type=float, default=1.0,
                    help="Temperature for sampling")
parser.add_argument("--top_k", type=int, default=0,
                    help="top_k for sampling (0 = disabled)")
parser.add_argument("--typical_p", type=float, default=None,
                    help="typical_p for typical decoding (None = disabled)")
parser.add_argument("--method", type=str, default="topk",
                    choices=["topk", "typical"],
                    help="Which sampling method to apply: 'topk' or 'typical'")
parser.add_argument("--device", type=str, default="cuda",
                    help="Device to use: 'cuda' or 'cpu'")
args = parser.parse_args()

# ------------------------------------------------------------------
# 2. Initialize device and random seed
# ------------------------------------------------------------------
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
print(f"Using device = {device}")

# ------------------------------------------------------------------
# 3. Build a GPT2LMHeadModel from default config
#    We'll load the EXACT shape from the checkpoint
# ------------------------------------------------------------------
# We do NOT from_pretrained(...) because that might mismatch shapes
# We'll just construct an empty GPT2LMHeadModel and fill weights
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")
model.eval()
model.to(device)

# ------------------------------------------------------------------
# 4. Load your checkpoint
# ------------------------------------------------------------------
print(f"Loading checkpoint from {args.ckpt}")
ckpt = torch.load(args.ckpt, map_location=device)

# If your checkpoint is stored as {'model_state': state_dict, ...}:
if "model_state" in ckpt:
    state_dict = ckpt["model_state"]
else:
    state_dict = ckpt

unwanted_prefix = "_orig_mod."
for k in list(state_dict.keys()):
    if k.startswith(unwanted_prefix):
        newk = k[len(unwanted_prefix):]
        state_dict[newk] = state_dict.pop(k)

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Missing keys: {missing}")
print(f"Unexpected keys: {unexpected}")
print("Checkpoint loaded successfully.")

# ------------------------------------------------------------------
# 5. Prepare the BERT tokenizer (for uer/gpt2-distil-chinese)
# ------------------------------------------------------------------
print("Loading BertTokenizer for gpt2-distil-chinese-cluecorpussmall ...")
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")

# ------------------------------------------------------------------
# 6. Define sampling functions
# ------------------------------------------------------------------
def apply_top_k_filter(logits, top_k):
    """ Retain only top_k tokens with highest logits; set rest to -inf. """
    if top_k <= 0:
        return logits
    # find top_k
    values, indices = torch.topk(logits, k=top_k, dim=-1)
    min_values = values[:, -1].unsqueeze(-1)
    # set those below min_values to -inf
    logits[logits < min_values] = float('-inf')
    return logits

def apply_typical_filter(logits, typical_p=0.9, min_tokens_to_keep=1):
    """
    Simplified typical decoding filter, referencing HF's TypicalLogitsWarper
    """
    # compute log_softmax
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    # compute distribution entropy
    ent = -(log_probs * probs).sum(dim=-1, keepdim=True)

    # distance from distribution entropy
    shifted_scores = torch.abs((-log_probs) - ent)
    # sort ascending
    sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False, dim=-1)
    # gather original logits in that order
    sorted_logits = logits.gather(-1, sorted_indices)

    # compute cumsum of sorted_probs
    sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    # find cutoff
    # how many tokens are below typical_p mass
    cutoff_indices = (cum_probs < typical_p).sum(dim=-1)
    cutoff_indices.clamp_(max=logits.shape[-1] - 1)

    # we create a boolean mask
    to_remove = torch.zeros_like(sorted_scores, dtype=torch.bool)
    for b in range(logits.size(0)):
        c_idx = cutoff_indices[b].item()
        c_idx = max(c_idx, min_tokens_to_keep - 1)
        to_remove[b, c_idx+1:] = True

    # scatter back
    to_remove_original = to_remove.scatter(1, sorted_indices, to_remove)
    logits = logits.masked_fill(to_remove_original, float('-inf'))
    return logits

@torch.no_grad()
def generate(model, input_ids, max_new_tokens=50, temperature=1.0,
             method="topk", top_k=40, typical_p=0.9):
    # block_size (for GPT2LMHeadModel) often 1024
    block_size = model.config.n_positions
    for _ in range(max_new_tokens):
        if input_ids.size(1) >= block_size:
            x_cond = input_ids[:, -block_size:]
        else:
            x_cond = input_ids

        # forward
        outputs = model(x_cond)
        logits = outputs.logits[:, -1, :] / temperature

        if method == "topk":
            # apply top-k
            logits = apply_top_k_filter(logits, top_k)
        elif method == "typical":
            # apply typical decoding
            logits = apply_typical_filter(logits, typical_p=typical_p)

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=1)
    return input_ids

# ------------------------------------------------------------------
# 7. Encode the prompt
# ------------------------------------------------------------------
if not args.prompt:
    # If user didn't provide a prompt, default to your "2.1.3" lyrics
    # e.g. 
    prompt_str = (
        "你的柔情似水 ，几度让我爱得沉醉 ，毫无保留不知道后悔 ，"
        "你能不能体会真情可贵 ，没有余力伤悲 ，爱情像难收的覆水 ，"
        "长长来路走的太憔悴"
    )
else:
    prompt_str = args.prompt

input_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

# ------------------------------------------------------------------
# 8. Generate samples
# ------------------------------------------------------------------
print(f"=== Generating {args.num_samples} samples with method={args.method}, top_k={args.top_k}, typical_p={args.typical_p}, temperature={args.temperature} ===")

for i in range(args.num_samples):
    out_ids = generate(model, input_ids.clone(),
                       max_new_tokens=args.max_new_tokens,
                       temperature=args.temperature,
                       method=args.method,
                       top_k=args.top_k,
                       typical_p=args.typical_p)
    out_tokens = out_ids[0].tolist()
    gen_text = tokenizer.decode(out_tokens)
    print(f"[Sample #{i+1}]:\n{gen_text}\n{'-'*40}\n")
