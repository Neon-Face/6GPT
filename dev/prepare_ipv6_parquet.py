"""
6GPT Data Prep: Convert raw IPv6 text file to Parquet shards.
"""
import os
import pyarrow as pa
import pyarrow.parquet as pq
import random

# 1. 配置
INPUT_FILE = "data/ipv6_addresses.txt" # 你的原始数据
OUTPUT_DIR = "/Users/yourname/.cache/nanochat/base_data" # Mac上的缓存目录
ROW_GROUP_SIZE = 1024 # 每次读取的行数
DOCS_PER_SHARD = 100000 # 每个文件存多少个IP (根据你的内存调整)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. 读取并打乱
print(f"Reading {INPUT_FILE}...")
with open(INPUT_FILE, 'r') as f:
    # 假设每行一个IP，去掉换行符
    all_ips = [line.strip() for line in f if line.strip()]

print(f"Total IPs: {len(all_ips)}. Shuffling...")
random.seed(42)
random.shuffle(all_ips)

# 3. 分片写入
shard_index = 0
current_shard_docs = []

# 切分 Train/Val (例如留后 1000 个做验证)
split_index = len(all_ips) - 1000
train_ips = all_ips[:split_index]
val_ips = all_ips[split_index:]

def write_shard(docs, index, split_name="train"):
    filename = f"shard_{index:05d}.parquet"
    # 如果是 val 集，最好用特殊命名，或者遵循 nanochat 逻辑(最后一个是val)
    # 这里我们简单起见，遵循 nanochat 逻辑：所有的都是 shard_xxxxx
    # 但 nanochat 的 dataset.py 假设最后一个文件是 val。
    
    path = os.path.join(OUTPUT_DIR, filename)
    table = pa.Table.from_pydict({"text": docs}) # 必须叫 'text' 列
    pq.write_table(table, path, row_group_size=ROW_GROUP_SIZE)
    print(f"Wrote {path} ({len(docs)} IPs)")

# 写入训练集
for ip in train_ips:
    current_shard_docs.append(ip)
    
    # 攒够了一个 shard 或者 凑齐了 row_group 倍数
    if len(current_shard_docs) >= DOCS_PER_SHARD and len(current_shard_docs) % ROW_GROUP_SIZE == 0:
        write_shard(current_shard_docs, shard_index)
        current_shard_docs = []
        shard_index += 1

# 写入剩余的训练集 (如果有)
if current_shard_docs:
    # 补齐到 row_group_size (可选，但为了并行不出错最好补齐)
    remainder = len(current_shard_docs) % ROW_GROUP_SIZE
    if remainder != 0:
        padding = ["<|pad|>"] * (ROW_GROUP_SIZE - remainder) # 使用你的 pad token
        current_shard_docs.extend(padding)
    write_shard(current_shard_docs, shard_index)
    shard_index += 1

# 写入验证集 (作为最后一个 shard)
# 同样补齐
remainder = len(val_ips) % ROW_GROUP_SIZE
if remainder != 0:
    val_ips.extend(["<|pad|>"] * (ROW_GROUP_SIZE - remainder))
write_shard(val_ips, shard_index)

print("Done!")