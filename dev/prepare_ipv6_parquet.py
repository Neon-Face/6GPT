"""
Low Memory IPv6 Data Prep for MacBook M3 Air.
Strategy: Random Bucketing -> Local Dedup -> Parquet
"""

import os
import lzma
import random
import shutil
import ipaddress
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ================= CONFIGURATION =================
INPUT_FILE = "data/responsive-addresses.txt.xz"
OUTPUT_DIR = os.path.expanduser("~/.cache/nanochat/base_data")
TEMP_DIR = "data/temp_buckets" # 临时文件夹

# 将数据随机分散到多少个桶里。
# 1GB 压缩数据 -> 解压约 15GB。
# 分 100 个桶 -> 每个桶 150MB。Mac M3 处理起来轻轻松松。
NUM_BUCKETS = 100 

DOCS_PER_SHARD = 200000 
# =================================================

def expand_ipv6(ip_str):
    try:
        return ipaddress.IPv6Address(ip_str).exploded
    except ValueError:
        return None

def process_low_memory():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    # 1. 准备目录
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 打开所有桶的文件句柄
    print(f"Step 1: Streaming & Random Bucketing into {NUM_BUCKETS} files...")
    bucket_files = []
    try:
        # 创建 100 个临时文件
        for i in range(NUM_BUCKETS):
            f = open(os.path.join(TEMP_DIR, f"bucket_{i}.txt"), 'w')
            bucket_files.append(f)

        # 流式读取 xz
        with lzma.open(INPUT_FILE, mode='rt', encoding='utf-8') as fin:
            for line in tqdm(fin, desc="Distributing"):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # 随机选一个桶写入 (这实现了全局打乱！)
                # 我们先不展开，为了节省临时文件的写入体积和IO
                # 也不去重，留到桶内处理
                bucket_idx = random.randint(0, NUM_BUCKETS - 1)
                bucket_files[bucket_idx].write(line + '\n')

    finally:
        # 关闭所有文件
        for f in bucket_files:
            f.close()

    print("\nStep 2: Processing Buckets (Expand, Dedup, Parquet)...")
    
    shard_index = 0
    total_processed_ips = 0

    # 遍历每个桶
    for i in range(NUM_BUCKETS):
        bucket_path = os.path.join(TEMP_DIR, f"bucket_{i}.txt")
        if not os.path.exists(bucket_path):
            continue
            
        # 读取桶内容到内存 (此时只有总数据的 1/100，很小)
        with open(bucket_path, 'r') as f:
            raw_lines = f.readlines()
        
        # 桶内去重 & 展开
        unique_ips = set()
        for line in raw_lines:
            expanded = expand_ipv6(line.strip())
            if expanded:
                unique_ips.add(expanded)
        
        # 转为列表并打乱 (桶内打乱)
        # 此时 list 很小，shuffle 很快
        ip_list = list(unique_ips)
        random.shuffle(ip_list)
        
        # 写入 Parquet
        # 注意：这里我们可能一个桶生成多个 shard，或者多个桶凑一个 shard
        # 为简单起见，我们把桶里的数据按 DOCS_PER_SHARD 切分写入
        
        current_idx = 0
        while current_idx < len(ip_list):
            chunk = ip_list[current_idx : current_idx + DOCS_PER_SHARD]
            
            # 如果 chunk 太小（比如桶末尾只剩几个IP），其实可以直接写，
            # Dataset.py 能处理小文件。
            if len(chunk) > 0:
                table = pa.Table.from_pydict({"text": chunk})
                shard_name = f"shard_{shard_index:05d}.parquet"
                pq.write_table(table, os.path.join(OUTPUT_DIR, shard_name), compression='zstd')
                shard_index += 1
                total_processed_ips += len(chunk)
            
            current_idx += DOCS_PER_SHARD
            
        # 释放内存
        del unique_ips
        del ip_list
        print(f"Processed Bucket {i+1}/{NUM_BUCKETS} - Total IPs so far: {total_processed_ips:,}", end='\r')

    # 清理临时文件
    print(f"\nCleaning up temp files...")
    shutil.rmtree(TEMP_DIR)
    
    print(f"\nAll Done! Generated {shard_index} shards containing {total_processed_ips:,} IPs.")
    print(f"Data saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_low_memory()