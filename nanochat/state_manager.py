# nanochat/state_manager.py
import os
from pybloom_live import ScalableBloomFilter

class StateManager:
    """
    Implements the Dual Bloom Filter system described in the 6GPT methodology.
    1. probed_filter: Tracks ALL generated addresses (to penalize duplicates).
    2. active_filter: Tracks confirmed ACTIVE addresses (to prioritize exploitation).
    """
    def __init__(self, initial_capacity=100000, error_rate=0.001):
        # probed_filter: 这里的地址如果我们再生成，就给负分（惩罚重复）
        self.probed_filter = ScalableBloomFilter(initial_capacity=initial_capacity, error_rate=error_rate)
        
        # active_filter: 这里的地址是我们的战利品
        self.active_filter = ScalableBloomFilter(initial_capacity=initial_capacity/10, error_rate=error_rate)
        
        self.total_generated = 0
        self.total_unique = 0
        self.total_active = 0

    def check_and_add(self, ip_list, active_status_list):
        """
        输入: 
            ip_list: List[str] generated IPs
            active_status_list: List[float] 1.0 or 0.0
        
        返回: 
            rewards: List[float]
            is_novel: List[bool]
        """
        rewards = []
        is_novel_list = []

        for ip, is_active in zip(ip_list, active_status_list):
            # 1. Check Probed Filter (Novelty Check)
            if ip in self.probed_filter:
                # 重复生成的惩罚 (Penalty for redundancy)
                # 即使它是活的，如果已经发现过了，也稍微惩罚一下，迫使模型去探索新的
                reward = -0.5 
                is_novel = False
            else:
                # 新发现的地址！
                self.probed_filter.add(ip)
                self.total_generated += 1
                self.total_unique += 1
                is_novel = True
                
                if is_active > 0.5:
                    # 这是一个全新的、活的 IP！大奖！
                    reward = 1.0
                    self.active_filter.add(ip)
                    self.total_active += 1
                else:
                    # 全新，但是死的。小惩罚，或者不惩罚（鼓励探索）
                    reward = -0.1 

            rewards.append(reward)
            is_novel_list.append(is_novel)
            
        return rewards, is_novel_list

    def get_stats(self):
        return {
            "total_generated": self.total_generated,
            "unique_probing": self.total_unique,
            "confirmed_active": self.total_active,
            "hit_rate": (self.total_active / self.total_unique) if self.total_unique > 0 else 0
        }