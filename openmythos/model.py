import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) for memory-efficient attention.
    Compresses KV cache into a latent vector.
    """
    def __init__(self, dim: int, num_heads: int, latent_dim: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.latent_dim = latent_dim

        # Compression
        self.q_proj = nn.Linear(dim, dim)
        self.kv_compress = nn.Linear(dim, latent_dim)
        self.kv_up = nn.Linear(latent_dim, dim * 2) # For K and V

        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, latent_kv: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, c = x.size()
        
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        
        if latent_kv is None:
            latent_kv = self.kv_compress(x)
        
        kv = self.kv_up(latent_kv).view(b, t, 2, self.num_heads, self.head_dim)
        k = kv[:, :, 0].transpose(1, 2)
        v = kv[:, :, 1].transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.out_proj(out), latent_kv

class MythosBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, latent_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadLatentAttention(dim, num_heads, latent_dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x: torch.Tensor, latent_kv: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h, lkv = self.attn(self.ln1(x), latent_kv)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x, lkv

class OpenMythos(nn.Module):
    """
    OpenMythos: Recurrent-Depth Transformer.
    """
    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 16,
        latent_dim: int = 128,
        num_prelude_layers: int = 4,
        num_recurrent_layers: int = 8,
        num_coda_layers: int = 4,
        max_loop_iters: int = 16,
        vocab_size: int = 50257
    ):
        super().__init__()
        self.dim = dim
        self.max_loop_iters = max_loop_iters
        
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # Prelude
        self.prelude = nn.ModuleList([
            MythosBlock(dim, num_heads, latent_dim) for _ in range(num_prelude_layers)
        ])
        
        # Recurrent Block
        self.recurrent_block = nn.ModuleList([
            MythosBlock(dim, num_heads, latent_dim) for _ in range(num_recurrent_layers)
        ])
        
        # Coda
        self.coda = nn.ModuleList([
            MythosBlock(dim, num_heads, latent_dim) for _ in range(num_coda_layers)
        ])
        
        self.head = nn.Linear(dim, vocab_size)

    @tracer.start_as_current_span("openmythos_forward")
    def forward(self, tokens: torch.Tensor, loop_iters: Optional[int] = None) -> torch.Tensor:
        if loop_iters is None:
            loop_iters = self.max_loop_iters
            
        x = self.embedding(tokens)
        
        # Prelude
        for block in self.prelude:
            x, _ = block(x)
            
        # Input Injection: Save the state after prelude
        prelude_state = x
        
        # Recurrent Looping
        for i in range(loop_iters):
            with tracer.start_as_current_span(f"recurrent_loop_{i}"):
                # Inject prelude state (optional/additive)
                x = x + prelude_state 
                
                for block in self.recurrent_block:
                    x, _ = block(x)
                    
        # Coda
        for block in self.coda:
            x, _ = block(x)
            
        return self.head(x)

if __name__ == "__main__":
    # Test instantiation
    model = OpenMythos(
        dim=256,
        num_prelude_layers=2,
        num_recurrent_layers=4,
        num_coda_layers=2,
        max_loop_iters=4
    )
    test_input = torch.randint(0, 50257, (1, 32))
    output = model(test_input)
    print(f"Output shape: {output.shape}")
