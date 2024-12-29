import os
import json
import pickle
import numpy as np
import sentencepiece as spm  # pip install sentencepiece

# 1. 指定数据路径
data_file_path = "/home/v-zhifeng/HPE/nanoGPT/lyric_data_for_CL_no_id.jsonl"

# 2. 加载你的 SentencePiece 模型
sp = spm.SentencePieceProcessor()
sp.load("/home/v-zhifeng/HPE/nanoGPT/lyrics.model")

# 3. 初始化容器，用于收集 train/valid/test 歌词
train_lyrics = []
valid_lyrics = []
test_lyrics = []

# 4. 读取数据并按照 dataset_split 分三份
with open(data_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        if entry["dataset_split"] == "train":
            train_lyrics.extend(entry["lyric"])
        elif entry["dataset_split"] == "valid":
            valid_lyrics.extend(entry["lyric"])
        elif entry["dataset_split"] == "test":
            test_lyrics.extend(entry["lyric"])

# 5. 合并为字符串（用换行符拼接）
train_data = "\n".join(train_lyrics)
valid_data = "\n".join(valid_lyrics)
test_data = "\n".join(test_lyrics)

print(f"Train data length (in characters) : {len(train_data):,}")
print(f"Valid data length (in characters) : {len(valid_data):,}")
print(f"Test data length  (in characters) : {len(test_data):,}")

# ----------------------------------------------------------------------------
# 6. 使用 SentencePiece 对文本进行分词 (BPE)
#    sp.encode() 支持对字符串进行 BPE 分词，返回token列表
# ----------------------------------------------------------------------------
# 说明: 这里演示“整段文本”一次性 encode。如果数据特别大，可考虑分块处理
train_ids = sp.encode(train_data, out_type=int)
valid_ids = sp.encode(valid_data, out_type=int)
test_ids = sp.encode(test_data, out_type=int)

# 7. 输出统计信息
vocab_size = sp.vocab_size()  # 来自你的 lyrics.model
print(f"Vocabulary size (SentencePiece BPE): {vocab_size}")
print(f"Train dataset tokens:  {len(train_ids):,}")
print(f"Valid dataset tokens:  {len(valid_ids):,}")
print(f"Test dataset tokens:   {len(test_ids):,}")

# ----------------------------------------------------------------------------
# 8. 将 token 序列转换为 numpy 并写入二进制文件 (.bin)
# ----------------------------------------------------------------------------
# 如果你的 vocab_size < 65536，且 token 不会超过 65535，可以用 np.uint16
# 否则用 np.uint32 更安全
train_arr = np.array(train_ids, dtype=np.uint32)
valid_arr = np.array(valid_ids, dtype=np.uint32)
test_arr = np.array(test_ids, dtype=np.uint32)

output_dir = os.path.dirname(data_file_path)
train_bin_path = os.path.join(output_dir, 'train.bin')
valid_bin_path = os.path.join(output_dir, 'val.bin')
test_bin_path = os.path.join(output_dir, 'test.bin')

train_arr.tofile(train_bin_path)
valid_arr.tofile(valid_bin_path)
test_arr.tofile(test_bin_path)

# ----------------------------------------------------------------------------
# 9. 保存 meta 信息到 meta.pkl
#    这里没有 stoi/itos，因为我们用的是 SentencePiece。
#    只需要记住 vocab_size 和 model 文件路径即可。
# ----------------------------------------------------------------------------
meta = {
    'vocab_size': vocab_size,
    'sentencepiece_model': "/home/v-zhifeng/HPE/nanoGPT/lyrics.model",
    # 你可在后续推理脚本里再用同一份model加载spm
}

with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Data preparation complete.")
print(f"Saved: {train_bin_path}, {valid_bin_path}, {test_bin_path}, and meta.pkl.")
