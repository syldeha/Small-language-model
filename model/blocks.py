from model.attention import GroupQueryAttention, FlashGroupQueryAttention
from model.FFN import FeedForward, RMSNorm
import torch.nn as nn

# def __init__(self, d_in,num_kv_groups, num_heads=None,head_dim=None,qk_norm=False,dtype=None):
class TransformerBlock(nn.Module) : 
    def __init__(self, cfg) : 
        super().__init__()
        self.att=FlashGroupQueryAttention(
            d_in=cfg['emb_dim'], 
            num_kv_groups=cfg["n_kv_groups"], 
            num_heads=cfg['n_heads'],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"],
            head_dim=cfg['head_dim'],
        )

        self.ff=FeedForward(cfg)
        self.norm1=RMSNorm(cfg['emb_dim'], eps=1e-6)
        self.norm2=RMSNorm(cfg['emb_dim'], eps=1e-6)


    def forward(self, x, mask , cos, sin, start_pos=0, cache=None) : 
        shortcut=x
        x=self.norm1(x)
        x,next_cache=self.att(
            x, mask, cos, sin, start_pos=start_pos, cache=cache
        )
        x=x+shortcut
        shortcut=x

        x=self.norm2(x)
        x=self.ff(x)
        x=x+shortcut

        return x, next_cache
