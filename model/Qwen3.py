import torch 
import torch.nn as nn 
from model.blocks import TransformerBlock
from model.FFN import RMSNorm
from model.attention import compute_rope_params
import math

class Qwen3Model(nn.Module) : 
    def __init__(self, cfg) : 
        super().__init__()
        self.tok_emb=torch.nn.Embedding(cfg["vocab_size"] , cfg["emb_dim"], dtype=cfg['dtype'])
        self.trf_blocks=torch.nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm=RMSNorm(cfg["emb_dim"])
        self.out_head=torch.nn.Linear(cfg["emb_dim"], cfg["vocab_size"],bias=False, dtype=cfg["dtype"])

        #utilities 
        if cfg["head_dim"] is None :
            head_dim=cfg["emb_dim"] // cfg["n_heads"]

        else : 
            head_dim=cfg["head_dim"] 


        cos, sin=compute_rope_params(
            head_dim=head_dim, 
            theta_base=cfg["rope_base"], 
            context_length=cfg["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.cfg=cfg
        self.current_pos=0

        self.apply(self._init_weights)


        for pn , p in self.named_parameters() : 
            if pn.endswith("out_proj.weight") : 
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*cfg['n_layers']))

    def _init_weights(self, module) : 
        if isinstance(module, nn.Linear) : 
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None : 
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding): 
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)



    def forward(self, in_idx, cache=None) : 
        tok_embs=self.tok_emb(in_idx)
        x=tok_embs

        num_tokens=x.shape[1]
        if cache is not None :
            pos_start=self.current_pos
            pos_end=self.current_pos+num_tokens

            mask=torch.triu(
                torch.ones(pos_end, pos_end, device=x.device,dtype=torch.bool), 
                diagonal=1
            )[pos_start:pos_end , pos_end]
            self.current_pos=pos_end
        else : 
            pos_start=0
            mask=torch.triu(
                torch.ones(num_tokens, num_tokens, device=x.device,dtype=torch.bool),
                diagonal=1
            )

        mask=mask[None , None,:,:]

        for idx, block in enumerate(self.trf_blocks) : 
            blk_cache=cache.get(idx) if cache else None
            x,new_blk_cache=block(x, mask, self.cos, self.sin, start_pos=pos_start, cache=blk_cache)

            if cache is not None : 
                cache.update(idx, new_blk_cache)
        x=self.final_norm(x)

        logits=self.out_head(x.to(self.cfg["dtype"]))
        return logits
        
    def reset_kv_cache (self):
        self.current_pos=0
    