# config/6gpt_mac.py

# 1. 硬件适配
device_type = 'mps'  # Apple Silicon 加速
grad_clip = 1.0

# 2. 模型架构 (轻量级)
# depth=6 => 6层 transformer
# model_dim = 6 * 64 = 384 维度
depth = 6
max_seq_len = 16     # 极其重要！节省大量内存
vocab_size = 65543   # 你的tokenizer大小 (约数，脚本会自动对齐)

# 3. 训练参数
# Mac 内存有限，device_batch_size 开小点
device_batch_size = 64 
# total_batch_size 稍微大点，保证训练稳定
total_batch_size = 32768 # 64 * 16 * 32 (accum steps)

# 4. 学习率 (由于是全新词表，可以稍微大一点)
learning_rate = 1e-3 

# 5. 周期
num_iterations = 2000 # 先跑 2000 步看看效果
eval_every = 100
sample_every = 200
save_every = 1000

# 6. 数据
# 确保你的 dataset.py 指向了正确的数据目录