"""
IPv6 Segment Tokenizer for NanoChat.
Maps each 16-bit segment (hextet) of an IPv6 address to a token ID (0-65535).
"""

import os
import ipaddress
import torch
import copy
from functools import lru_cache

# -----------------------------------------------------------------------------
# Configuration

# 0-65535 are reserved for the actual hex values of the IPv6 segments.
# We start special tokens from 65536.
TOKEN_OFFSET = 65536 

SPECIAL_TOKENS = [
    "<|bos|>",          # 65536: Beginning of Sequence
    "<|eos|>",          # 65537: End of Sequence (Crucial for generation stop)
    "<|pad|>",          # 65538: Padding
    # Chat/RL specific tokens (kept for compatibility with chat_rl.py)
    "<|user_start|>", 
    "<|user_end|>",
    "<|assistant_start|>", 
    "<|assistant_end|>",
]

class IPv6SegmentTokenizer:
    """
    Tokenizer that treats every IPv6 hextet (16-bit segment) as a single token.
    Vocabulary Size = 65536 + len(SPECIAL_TOKENS).
    Sequence Length = 8 (plus BOS/EOS).
    """

    def __init__(self):
        # Build the special tokens map
        self.special_tokens_map = {}
        self.id_to_special_token = {}
        
        for i, token in enumerate(SPECIAL_TOKENS):
            token_id = TOKEN_OFFSET + i
            self.special_tokens_map[token] = token_id
            self.id_to_special_token[token_id] = token

        self.bos_token_id = self.special_tokens_map["<|bos|>"]
        self.eos_token_id = self.special_tokens_map["<|eos|>"]
        self.pad_token_id = self.special_tokens_map["<|pad|>"]
        
        # Vocab size = Real values + Special tokens
        self.vocab_size = TOKEN_OFFSET + len(SPECIAL_TOKENS)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        # Since this is a rule-based tokenizer, we don't really need to load anything.
        # But we implement this to satisfy the NanoChat interface.
        return cls()

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # This tokenizer is fixed rule-based, training is a no-op.
        print("IPv6SegmentTokenizer uses a fixed vocabulary. Training skipped.")
        return cls()

    def get_vocab_size(self):
        return self.vocab_size

    def get_special_tokens(self):
        return self.special_tokens_map

    def id_to_token(self, idx):
        """Debug function to print meaningful string for an ID"""
        if idx in self.id_to_special_token:
            return self.id_to_special_token[idx]
        if 0 <= idx <= 65535:
            # Return hex format, e.g., "0db8" or "0000"
            return f"{idx:04x}"
        return ""

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.special_tokens_map.get(text, self.eos_token_id)

    def get_bos_token_id(self):
        return self.bos_token_id

    def _encode_one(self, text, prepend=None, append=None):
        """
        Core logic:
        1. Expand IPv6 (2001:db8::1 -> 2001:0db8:0000:0000:0000:0000:0000:0001)
        2. Split by ':'
        3. Convert hex string to int (0000 -> 0, ffff -> 65535)
        """
        ids = []
        
        # Handle Prepend (usually BOS)
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)

        # Handle Body
        if text.strip(): # if text is not empty
            try:
                # ipaddress library handles expansion perfectly
                # It handles "::" automatically
                addr = ipaddress.IPv6Address(text.strip())
                full_str = addr.exploded # e.g. "2001:0db8:..."
                
                # Split into 8 segments
                parts = full_str.split(':')
                
                # Convert hex to int indices
                # Note: '0000' becomes integer 0. This fulfills your requirement 
                # effectively, as 0 is a unique token ID for empty segments.
                token_ids = [int(part, 16) for part in parts]
                ids.extend(token_ids)
                
            except ValueError:
                # Fallback for non-IP text (e.g. system prompts in RL)
                # Since we can only encode IPs, we might ignore or warn
                # For now, let's ignore to prevent crashing on system messages
                pass

        # Handle Append (usually EOS)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)

        return ids

    def encode(self, text, prepend=None, append=None, num_threads=1):
        # Support batch or single string
        if isinstance(text, str):
            return self._encode_one(text, prepend, append)
        elif isinstance(text, list):
            return [self._encode_one(t, prepend, append) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        """
        Convert IDs back to IPv6 string.
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
            
        parts = []
        for idx in ids:
            if idx in self.id_to_special_token:
                continue # Skip special tokens in output string
            if 0 <= idx <= 65535:
                parts.append(f"{idx:04x}")
        
        if len(parts) == 0:
            return ""
            
        full_ip = ":".join(parts)
        
        # Optional: Compress it back (e.g. restore ::) for readability
        try:
            return str(ipaddress.IPv6Address(full_ip))
        except ValueError:
            return full_ip # Return partial/invalid string if needed

    def save(self, tokenizer_dir):
        # Nothing to save, just create the dir
        os.makedirs(tokenizer_dir, exist_ok=True)
        # Create a dummy file just so NanoChat thinks it exists
        with open(os.path.join(tokenizer_dir, "ipv6_tokenizer_meta.txt"), "w") as f:
            f.write("Fixed Vocabulary IPv6 Tokenizer")
        print(f"Saved (Dummy) tokenizer to {tokenizer_dir}")

    def render_conversation(self, conversation, max_tokens=2048):
        """
        Adapted for 6GPT. We mostly ignore the 'Role' structure and just 
        extract the IPv6 address from the assistant or user content.
        """
        ids = []
        mask = [] # 1 for tokens to predict (assistant), 0 for others

        bos = self.get_bos_token_id()
        ids.append(bos)
        mask.append(0)

        for message in conversation["messages"]:
            content = message["content"]
            role = message["role"]
            
            # Encode the IP
            # We assume content contains JUST the IP address string
            encoded_ip = self.encode(content, prepend=None, append=None)
            
            if role == "user":
                # User provided IP (e.g. seed)? We don't train on it usually, 
                # but in Scanner context, maybe we treat everything as data?
                # Let's stick to standard SFT logic: User = Mask 0
                ids.extend(encoded_ip)
                mask.extend([0] * len(encoded_ip))
            
            elif role == "assistant":
                # Assistant generated IP -> Train on this!
                ids.extend(encoded_ip)
                mask.extend([1] * len(encoded_ip))
                
                # Add EOS for assistant
                ids.append(self.eos_token_id)
                mask.append(1)

        # Truncate
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask
        
    def render_for_completion(self, conversation):
        """
        Used in RL. We just want the model to complete the IP.
        """
        # Extract the last "prompt" (e.g. a prefix)
        # For a scanner, we might just give BOS and let it hallucinate from scratch.
        ids = [self.get_bos_token_id()]
        return ids

# -----------------------------------------------------------------------------
# Helper to replace the one in existing tokenizer.py

def get_tokenizer():
    # Return our new class directly
    return IPv6SegmentTokenizer()