import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, BertTokenizer

# -----------------------------------------------------------------------
# 1. Model / tokenizer setup (loaded once at app start)
# -----------------------------------------------------------------------
CKPT_PATH = "/home/v-zhifeng/HPE/nanoGPT/out-lyric-gpt2-distil-chinese/ckpt.pt"
MODEL_NAME = "uer/gpt2-distil-chinese-cluecorpussmall"  # The base GPT2-distil-chinese

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading base GPT2-distil-chinese config from:", MODEL_NAME)
hf_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
hf_model.config.tie_word_embeddings = False  # must match training script if you disabled tying
hf_model.eval().to(device)

print("Loading checkpoint from:", CKPT_PATH)
checkpoint = torch.load(CKPT_PATH, map_location=device)
if "model_state" in checkpoint:  # or similar keys
    state_dict = checkpoint["model_state"]
else:
    state_dict = checkpoint

# Remove unwanted prefix if any
unwanted_prefix = "_orig_mod."
for k in list(state_dict.keys()):
    if k.startswith(unwanted_prefix):
        new_k = k[len(unwanted_prefix):]
        state_dict[new_k] = state_dict.pop(k)

missing, unexpected = hf_model.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)
print("Checkpoint loaded!")

print("Loading BertTokenizer for GPT2-distil-chinese-cluecorpussmall ...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# -----------------------------------------------------------------------
# 2. Utility: apply top-k / typical-p
# -----------------------------------------------------------------------
def apply_top_k_filter(logits, top_k):
    if top_k <= 0:
        return logits
    values, indices = torch.topk(logits, top_k, dim=-1)
    min_values = values[:, -1].unsqueeze(-1)
    logits[logits < min_values] = float("-inf")
    return logits

def apply_typical_filter(logits, typical_p=0.9, min_tokens_to_keep=1):
    # compute log_softmax
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    # distribution entropy
    ent = -(log_probs * probs).sum(dim=-1, keepdim=True)

    shifted_scores = torch.abs((-log_probs) - ent)
    sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False, dim=-1)
    sorted_logits = logits.gather(-1, sorted_indices)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

    # find cutoff
    cutoff_ind = (cumsum_probs < typical_p).sum(dim=1)
    cutoff_ind.clamp_(max=logits.size(1) - 1)

    to_remove = torch.zeros_like(sorted_scores, dtype=torch.bool)
    for b in range(logits.size(0)):
        cix = cutoff_ind[b].item()
        cix = max(cix, min_tokens_to_keep - 1)
        to_remove[b, cix+1:] = True

    to_remove_original = to_remove.scatter(1, sorted_indices, to_remove)
    logits = logits.masked_fill(to_remove_original, float("-inf"))
    return logits


@torch.no_grad()
def generate_text(prompt, max_new_tokens=50, temperature=1.0,
                  top_k=0, typical_p=None):
    """
    Generate text from model, given a Chinese prompt and sampling params.
    """
    # encode prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)

    block_size = hf_model.config.n_positions if hasattr(hf_model.config, "n_positions") else 1024

    for _ in range(max_new_tokens):
        if input_ids.size(1) >= block_size:
            x_cond = input_ids[:, -block_size:]
        else:
            x_cond = input_ids

        outputs = hf_model(x_cond)
        logits = outputs.logits[:, -1, :] / temperature

        # typical-p
        if typical_p is not None and typical_p > 0 and typical_p < 1:
            logits = apply_typical_filter(logits, typical_p)

        # top-k
        if top_k > 0:
            logits = apply_top_k_filter(logits, top_k)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=1)

    # decode
    out_tokens = input_ids[0].tolist()
    return tokenizer.decode(out_tokens)

# -----------------------------------------------------------------------
# 3. Define Gradio UI
# -----------------------------------------------------------------------
import gradio as gr

def infer(prompt, max_new_tokens, temperature, top_k, typical_p):
    """Wrap the generate_text() function for Gradio"""
    # Call generation
    generated = generate_text(
        prompt, max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=int(top_k),
        typical_p=float(typical_p) if typical_p else None
    )
    return generated

with gr.Blocks() as demo:
    gr.Markdown("## GPT2-distil-chinese Demo with top-k & typical-p")

    with gr.Row():
        prompt = gr.Textbox(label="请输入你的中文prompt (UTF-8)", lines=5, 
                            value="你的柔情似水 ，几度让我爱得沉醉 ，毫无保留不知道后悔...")

    with gr.Row():
        max_new_tokens = gr.Slider(label="max_new_tokens", minimum=10, maximum=512, value=100, step=1)
        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.05)
    
    with gr.Row():
        top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=40, step=1)
        typical_p = gr.Slider(label="typical_p", minimum=0.0, maximum=1.0, value=0.0, step=0.01)

    generate_btn = gr.Button("Generate")

    output = gr.Textbox(label="Generated Lyrics", lines=10)

    # Link the function
    def on_click_generate(prompt, max_new_tokens, temperature, top_k, typical_p):
        return infer(prompt, max_new_tokens, temperature, top_k, typical_p)

    generate_btn.click(
        fn=on_click_generate,
        inputs=[prompt, max_new_tokens, temperature, top_k, typical_p],
        outputs=[output],
    )

demo.launch(server_name="0.0.0.0", share=False)
