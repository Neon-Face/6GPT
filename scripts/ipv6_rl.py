"""
6GPT Reinforcement Learning Loop.
Uses GRPO (Group Relative Policy Optimization) simplified style.

Cycle:
1. Generate batch of IPs.
2. Verify with ZMap/Scanner.
3. Compute Rewards (Surprisal + Novelty + Active).
4. Update Policy.
"""

import os
import torch
import torch.nn.functional as F
import wandb
import time

from contextlib import nullcontext 
from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb, autodetect_device_type 
from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.engine import Engine
from nanochat.scanner import IPv6Scanner
from nanochat.state_manager import StateManager

# ================= Configuration =================
# 实验名称
run = "6gpt_rl_exp1" 

# 基础模型来源 (d6 = depth 6)
# 注意：你需要手动指定 checkpoint 路径，或者让 load_model 自动找
# 建议：先把 base_checkpoints/d6/step_00xxx.pt 复制一份改为 base_checkpoints/d6/model.pt
# 或者修改下面的 load_logic
source = "d6" 

# 训练参数
device_batch_size = 128  # 每次生成多少个 IP
num_steps = 1000         # RL 循环多少次
learning_rate = 1e-5     # RL 需要很小的 LR，防止破坏 Base Model 的语法能力
entropy_coef = 0.01      # 熵正则化系数 (鼓励探索，防止 Mode Collapse)

# 奖励权重
W_ACTIVE = 10.0          # 发现活 IP 的奖励
W_NOVEL = 1.0            # 发现新 IP 的奖励
W_REPEAT = -0.5          # 重复生成的惩罚
W_SURPRISAL = 0.1        # 惊奇度奖励系数

# 生成参数
max_new_tokens = 10      
temperature = 1.2        # 初始温度设高点，鼓励探索

# =================================================

# 1. 初始化环境 (修改后支持 Mac M3)
device_type = autodetect_device_type() # 自动识别 mps
print0(f"Device Type Detected: {device_type}")

ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

# Mac M3 (MPS) 不需要显式的 autocast context，或者使用 nullcontext
if device_type == "cuda":
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
else:
    # 对于 MPS/CPU，直接通过
    autocast_ctx = nullcontext()
autocast_ctx = torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16)

# WandB
wandb_run = DummyWandb() if (run == "dummy" or not master_process) else wandb.init(project="6gpt", name=run)

# 2. 加载模型 & 裁判
print0(f"Loading Base Model: {source}...")
# 这里的 source 参数可能需要根据 checkpoint_manager 的逻辑调整
# 如果报错，请直接传入 pt 文件的绝对路径
try:
    model, tokenizer, meta = load_model(source, device, phase="train") 
except:
    # Fallback: 手动加载最近的 checkpoint
    print0("Standard load failed, trying manual path...")
    from nanochat.gpt import GPT, GPTConfig
    # 配置必须与 base_train 一致
    config = GPTConfig(vocab_size=65545, n_layer=6, n_head=6, n_embd=384, sequence_len=32)
    model = GPT(config)
    
    # === 修改开始 ===
    # 1. 指向正确的文件名 (model_000500.pt)
    ckpt_path = os.path.join(get_base_dir(), "base_checkpoints", "d6", "model_000500.pt")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    print0(f"Loading weights from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    
    # 2. 直接加载 state_dict (因为 model_000500.pt 里直接就是参数，没有嵌套 'model' key)
    # 如果报错 "missing keys"，可以尝试 print(state_dict.keys()) 看看结构，
    # 但通常 model_xxxxx.pt 就是纯参数。
    if "model" in state_dict:
        model.load_state_dict(state_dict['model'], strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    # === 修改结束 ===

    model.to(device)
    from nanochat.tokenizer import get_tokenizer
    tokenizer = get_tokenizer()

# 冻结部分参数？不，我们全量微调，但 LR 很小。
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
engine = Engine(model, tokenizer)

# 初始化 Scanner 和 StateManager
scanner = IPv6Scanner(use_mock=True) # ⚠️ DEBUG模式: 先用 Mock 跑通流程，确认无误后再改为 False
state_manager = StateManager()

print0("System Ready. Starting RL Loop...")

# ================= RL Loop =================

for step in range(num_steps):
    t0 = time.time()
    
    # --- A. Generate (生成) ---
    # 提示词：我们可以给它一些前缀，也可以让它从 BOS 开始
    # 这里我们随机给一些常用前缀，或者空前缀
    prefixes = ["<|bos|>"] * device_batch_size
    tokens = [tokenizer.encode(p, prepend=None) for p in prefixes]
    
    # 这一步需要 careful，engine.generate_batch 输入需要是对齐的 list
    # 简单起见，我们假设 engine 支持 batch 处理
    # 如果不支持，我们需要手动 padding。但 NanoChat engine 应该支持。
    # 这里我们简化：所有 prompt 都是 BOS (id=65536)
    bos_id = tokenizer.get_bos_token_id()
    prompt = [bos_id] 
    
    model.eval()
    with torch.no_grad():
        # 生成 IP
        # num_samples=device_batch_size 表示我们从这同一个种子并行生成 128 个不同的结果
        gen_tokens, _ = engine.generate_batch(
            prompt, 
            num_samples=device_batch_size, 
            max_tokens=max_new_tokens, 
            temperature=temperature
        )

    # 解码 IP 字符串
    ip_list = []
    valid_indices = [] # 记录哪些是合法 IP (能被 scanner 识别)
    
    for i, seq in enumerate(gen_tokens):
        # 去掉 BOS
        content = seq[1:] 
        ip_str = tokenizer.decode(content)
        # 简单的合法性检查
        if ":" in ip_str and len(ip_str) > 5:
            ip_list.append(ip_str)
            valid_indices.append(i)
        else:
            # 生成了垃圾 (比如空字符串)
            pass

    if len(ip_list) == 0:
        print0("Warning: Model generated 0 valid IPs this batch.")
        continue

    # --- B. Verify (验证) ---
    # 调用 Scanner
    scan_results = scanner.verify_batch(ip_list) # [0.0, 1.0, ...]
    
    # --- C. Reward (奖励计算) ---
    # 调用 Bloom Filter
    # bloom_rewards: [1.0, -0.5, ...]
    bloom_rewards, is_novel = state_manager.check_and_add(ip_list, scan_results)
    
    final_rewards = []
    for r in bloom_rewards:
        final_rewards.append(r * W_ACTIVE if r > 0 else r) # 放大发现活跃IP的奖励

    # --- D. Update (PPO/REINFORCE Update) ---
    # 我们需要重新计算 log_probs (这是 RL 的标准做法：再跑一次 Forward)
    
    model.train()
    optimizer.zero_grad()
    
    # 准备 Training Batch
    # 我们只训练那些生成了有效字符串的样本
    # 将 list 转为 tensor
    # 需要 padding! 
    max_len = max([len(gen_tokens[i]) for i in valid_indices])
    padded_inputs = []
    padded_targets = []
    
    for i, idx in enumerate(valid_indices):
        seq = gen_tokens[idx]
        # Pad with EOS or 0
        pad_len = max_len - len(seq)
        seq_tensor = seq + [0] * pad_len # 0 is safe padding
        
        # Input: [BOS, t1, t2, ...]
        # Target: [t1, t2, ..., EOS]
        # 但这里我们简化：Target 就是 Input 左移
        inp = torch.tensor(seq_tensor[:-1], dtype=torch.long, device=device)
        tgt = torch.tensor(seq_tensor[1:], dtype=torch.long, device=device)
        # Mask padding in target
        if pad_len > 0:
            tgt[-pad_len:] = -1
            
        padded_inputs.append(inp)
        padded_targets.append(tgt)
        
    inputs = torch.stack(padded_inputs)
    targets = torch.stack(padded_targets)
    rewards_tensor = torch.tensor(final_rewards, dtype=torch.float, device=device)
    
    # Forward Pass
    # NanoChat model forward 返回 logits 或 loss
    # 我们需要 logits 来算 log_probs
    logits = model(inputs) # [B, T, V]
    
    # 计算 Cross Entropy Loss (就是 log_probs 的负数)
    # loss_per_token: [B, T]
    loss_per_token = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none').view(inputs.size())
    
    # 策略梯度核心公式: Loss = - (Reward * LogProb)
    # LogProb = - CrossEntropy
    # 所以 Loss = Reward * CrossEntropy
    # 但我们要 minimize Loss，所以是 (Reward - Baseline) * CrossEntropy
    
    # Advantage Normalization (Baseline)
    adv = rewards_tensor - rewards_tensor.mean()
    
    # 加权 Loss
    # 我们只关心整个序列的平均 Loss
    seq_loss = loss_per_token.mean(dim=1) # [B]
    policy_loss = (seq_loss * adv).mean()
    
    # Entropy Regularization (鼓励探索)
    # 这是一个简化版的 entropy，直接基于 logits
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    
    total_loss = policy_loss - entropy_coef * entropy
    
    total_loss.backward()
    
    # Gradient Clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    dt = time.time() - t0
    
    # --- Logging ---
    stats = state_manager.get_stats()
    print0(f"Step {step}/{num_steps} | Loss: {total_loss.item():.4f} | Hit Rate: {stats['hit_rate']:.4f} | Found: {stats['confirmed_active']} | dt: {dt:.2f}s")
    
    wandb_run.log({
        "rl/loss": total_loss.item(),
        "rl/hit_rate": stats['hit_rate'],
        "rl/total_active": stats['confirmed_active'],
        "rl/entropy": entropy.item()
    })
    
    # Save Checkpoint
    if step > 0 and step % 100 == 0:
        save_path = os.path.join(get_base_dir(), "rl_checkpoints", run)
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_path, f"step_{step}.pt"))

print0("RL Training Completed.")