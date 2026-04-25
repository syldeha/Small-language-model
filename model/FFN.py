import torch
import torch.nn as nn

class RMSNorm(nn.Module) : 
    def __init__(self,
            emb_dim, 
            eps=1e-6, 
            bias=False, 
    ) : 
        super().__init__()
        self.scale=nn.Parameter(torch.ones(emb_dim))
        self.eps=eps
        self.shift = None if not bias else nn.Parameter(torch.zeros(emb_dim))


    def forward(self , x) : 
        # x shape= (Batch_size, len_seq,emb_dim)
        input_type=x.dtype
        x=x.to(torch.float32)

        variance=x.pow(2).mean(dim=-1, keepdim=True)
        norm_x=x*torch.rsqrt(variance+self.eps)*self.scale    #use the torch.rsqrt which means reciprocal root scare
        if self.shift is not None : 
            norm_x=norm_x+self.shift  
        return norm_x.to(input_type) #important to make sure to return the same type as input
    


class FeedForward(nn.Module) : 

    def __init__(self, cfg) : 
        super().__init__()
        self.fc1=nn.Linear(in_features=cfg["emb_dim"], out_features=cfg["hidden_dim"], bias=False ,dtype=cfg["dtype"])
        self.fc3=nn.Linear(in_features=cfg["hidden_dim"], out_features=cfg["emb_dim"] , bias=False ,dtype=cfg["dtype"])
        self.fc2=nn.Linear(in_features=cfg["emb_dim"] , out_features=cfg["hidden_dim"] , bias=False ,dtype=cfg["dtype"])
        self.dtype=cfg["dtype"]

    def forward(self, x): 
        #x shape=(batch_size ,seq_len, emb_dim)
        # x=x.to(self.dtype)

        gate=nn.functional.silu(self.fc1(x)) #(B,T,hidden_dim)
        content=self.fc2(x)
        output=self.fc3(gate*content)

        return output
    

