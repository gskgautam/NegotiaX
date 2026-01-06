import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MechanisticAlignmentCircuit(nn.Module):
    def __init__(self, hidden_size: int, head_dim: int = 64, mode: str = "standard"):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.mode = mode
        
        # 1. Axis-specific head representations (The "Memory" M)
        # We have 3 axes: Helpfulness, Harmlessness, Honesty
        # Shape: (3, hidden_size)
        if self.mode == "generic":
            # A1: No explicit H-axis structure. Just a generic bottleneck attention.
            # We can use learnable tokens, but not tied to axes.
            self.axis_embeddings = nn.Parameter(torch.randn(3, hidden_size) * 0.02)
        else:
            self.axis_embeddings = nn.Parameter(torch.randn(3, hidden_size) * 0.02)

        # Projections to create axis vectors from hidden states
        if self.mode == "generic":
             # Generic projections, not semantically tied to Help/Harm/Honest initially
             self.proj_help = nn.Linear(hidden_size, hidden_size)
             self.proj_harm = nn.Linear(hidden_size, hidden_size)
             self.proj_honest = nn.Linear(hidden_size, hidden_size)
        else:
            self.proj_help = nn.Linear(hidden_size, hidden_size)
            self.proj_harm = nn.Linear(hidden_size, hidden_size)
            self.proj_honest = nn.Linear(hidden_size, hidden_size)
        
        # 2. Mini self-attention projections
        # We project the axis embeddings into Q, K, V
        self.q_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, head_dim, bias=False)
        
        if self.mode == "frozen":
            # A2: Freeze Q/K/V matrices
            for param in self.q_proj.parameters(): param.requires_grad = False
            for param in self.k_proj.parameters(): param.requires_grad = False
            for param in self.v_proj.parameters(): param.requires_grad = False
        
        # Output projection to match hidden size
        self.out_proj = nn.Linear(3 * head_dim, hidden_size, bias=False)
        
    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        
        # 1. Construct Axis Vectors from Input
        w_help = self.proj_help(hidden_states)   # (B, S, H)
        w_harm = self.proj_harm(hidden_states)   # (B, S, H)
        w_honest = self.proj_honest(hidden_states) # (B, S, H)
        
        # Stack them: M = (B, S, 3, H)
        m = torch.stack([w_help, w_harm, w_honest], dim=2)
        
        # 2. Mini Self-Attention over the 3 axes
        # q: (B, S, 3, head_dim)
        q = self.q_proj(m)
        k = self.k_proj(m)
        v = self.v_proj(m)
        
        # Attention scores: (B, S, 3, 3)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Output: (B, S, 3, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        
        # 3. Final Combination
        # Flatten the 3 heads: (B, S, 3 * head_dim)
        concat_output = attn_output.view(batch_size, seq_len, -1)
        
        # Project back to hidden size
        final_output = self.out_proj(concat_output)
        
        return final_output
