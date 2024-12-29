import os
import json
import pickle
import numpy as np

from transformers import BertTokenizer

# 1. 指定数据及输出目录
data_file_path = "/home/v-zhifeng/HPE/nanoGPT/lyric_data_for_CL_no_id.jsonl"
output_dir = os.path.dirname(data_file_path)  # 与 data_file_path 同级

train_bin_path = os.path.join(output_dir, 'train.bin')
valid_bin_path = os.path.join(output_dir, 'val.bin')
test_bin_path = os.path.join(output_dir, 'test.bin')
meta_path = os.path.join(output_dir, 'meta.pkl')

# 2. 初始化 tokenizer
#    这里示例用 "uer/gpt2-distil-chinese-cluecorpussmall"
#    如果你换成别的模型，也要相应改掉字符串
model_name = "uer/gpt2-distil-chinese-cluecorpussmall"
tokenizer = BertTokenizer.from_pretrained(model_name)

# 3. 读取 JSONL 数据并按照 dataset_split 分三份
train_lyrics = []
valid_lyrics = []
test_lyrics = []

with open(data_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        if entry["dataset_split"] == "train":
            train_lyrics.extend(entry["lyric"])
        elif entry["dataset_split"] == "valid":
            valid_lyrics.extend(entry["lyric"])
        elif entry["dataset_split"] == "test":
            test_lyrics.extend(entry["lyric"])

# 4. 将每个 split 合并成大字符串（用换行拼接）
train_data = "\n".join(train_lyrics)
valid_data = "\n".join(valid_lyrics)
test_data = "\n".join(test_lyrics)

print(f"Train data length (in characters): {len(train_data):,}")
print(f"Valid data length (in characters): {len(valid_data):,}")
print(f"Test data length  (in characters): {len(test_data):,}")

# 5. 分词：将整段文本一次性 encode 为 token id
#    add_special_tokens=False 避免在开头/结尾自动添加 [CLS] 或 [SEP]
train_ids = tokenizer.encode(train_data, add_special_tokens=False)
valid_ids = tokenizer.encode(valid_data, add_special_tokens=False)
test_ids = tokenizer.encode(test_data, add_special_tokens=False)

print(f"Train dataset tokens: {len(train_ids):,}")
print(f"Valid dataset tokens: {len(valid_ids):,}")
print(f"Test dataset tokens : {len(test_ids):,}")

# 6. 转成 numpy array 并保存二进制 .bin
#    如果 vocab_size < 65536 且不会出现 >65535 的 token id，可用 np.uint16
#    但为了稳妥，这里用 np.uint32
train_arr = np.array(train_ids, dtype=np.uint32)
valid_arr = np.array(valid_ids, dtype=np.uint32)
test_arr = np.array(test_ids, dtype=np.uint32)

train_arr.tofile(train_bin_path)
valid_arr.tofile(valid_bin_path)
test_arr.tofile(test_bin_path)

print(f"Saved: {train_bin_path}, {valid_bin_path}, {test_bin_path}")

# 7. 生成 meta.pkl 并保存
#    这里不再保存 stoi/itos，因为我们使用 HuggingFace tokenizer
#    只需要在后续脚本里复用同一个 tokenizer
meta = {
    "tokenizer_name": model_name,
    "vocab_size": tokenizer.vocab_size,  # 也可以 tokenizer.vocab_size
}
with open(meta_path, 'wb') as f:
    pickle.dump(meta, f)

print("Data preparation complete.")
print(f"Saved meta info to {meta_path}.")
