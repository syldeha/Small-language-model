import torch
from model.FFN import RMSNorm
import torch.nn as nn
# from flash_attn import flash_attn_func,flash_attn_varlen_func

#approche_1
def compute_rope_params(head_dim, theta_base=10_000, context_length=4096 , dtype=torch.float32):

    assert head_dim % 2==0, "Embbedding dimension must be even"

    inv_freq=1/(theta_base**(torch.arange(0,head_dim, 2,dtype=dtype)[: head_dim//2].float()/head_dim)) #(head_dim//2, )

    positions=torch.arange(context_length, dtype=dtype) #(seq_len, )
    angles=positions[:,None]*inv_freq[None, :]
    angles=torch.cat([angles, angles],dim=1)   #split half method  #(seq_len, head_dim)
    
    cos=torch.cos(angles) #(seq_len , head_dim)
    sin=torch.sin(angles)

    return cos , sin


def apply_rope(x,cos, sin, offset=0) : 

    batch_size, num_heads, seq_len, head_dim=x.shape  
    assert head_dim %2 ==0 , "embedding dimension should be even"

    x1=x[...,:head_dim//2]#first half
    x2=x[...,head_dim//2:]  #second half

    cos=cos[offset:offset+seq_len,:].unsqueeze(0).unsqueeze(0) #(1,1,seq_len, head_dim)
    sin=sin[offset:offset+seq_len, :].unsqueeze(0).unsqueeze(0) #(1,1 , seq_len, head_dim)

    rotated=torch.cat((-x2,x1), dim=-1)

    x_rotated=(x*cos)+(rotated*sin)
    return x_rotated.to(dtype=x.dtype)


#approche 2

def compute_rope_params_2(head_dim, theta_base=10_000, context_length=4096 , dtype=torch.float32):

    assert head_dim % 2==0, "Embbedding dimension must be even"

    inv_freq=1/(theta_base**(torch.arange(0,head_dim, 2,dtype=dtype)[: head_dim//2].float()/head_dim)) #(head_dim//2, )

    positions=torch.arange(context_length, dtype=dtype) #(seq_len, )
    angles=positions[:,None]*inv_freq[None, :]

    cos=torch.cos(angles).repeat_interleave(2,dim=-1) #(seq_len , head_dim)
    sin=torch.sin(angles).repeat_interleave(2,dim=-1)

    return cos , sin


def apply_rope_2(x,cos, sin, offset=0) : 

    batch_size, num_heads, seq_len, head_dim=x.shape  
    assert head_dim %2 ==0 , "embedding dimension should be even"

    x1=x[...,0::2]#(x0,x2,...,)
    x2=x[...,1::2]  #(x1,x3,...,)

    cos=cos[offset:offset+seq_len,:].unsqueeze(0).unsqueeze(0) #(1,1,seq_len, head_dim)
    sin=sin[offset:offset+seq_len, :].unsqueeze(0).unsqueeze(0) #(1,1 , seq_len, head_dim)

    rotated=torch.stack([-x2,x1], dim=-1).flatten(-2)

    x_rotated=(x*cos)+(rotated*sin)
    return x_rotated.to(dtype=x.dtype)

class SinusoidalPositionalEncoding(nn.Module) : 
    def __init__(self, seq_len, n_dim, dtype) :
        super().__init__()
        pos=torch.arange(seq_len)[:,None]

        theta=torch.exp(torch.arange(0,n_dim,2 ,dtype=dtype)/n_dim)*-torch.log(torch.tensor(10000.0))
        pos_theta=self.pos/theta[None, :]
        positional_encode=torch.zeros(seq_len,n_dim)
        positional_encode[:,0::2]=torch.cos(pos_theta)
        positional_encode[:,1::2]=torch.sin(pos_theta)

        self.register_buffer('positional_encode', positional_encode)

    def forward(self, x , offset) : 
        seq_len, _,_=x.shape
        return self.positional_encode[offset:offset+seq_len]

class LearnPositionalEncoding(nn.Module) : 
    def __init__(self, n_dim,seq_len) : 
        super.__init__()
        self.Embedding_layer=nn.Embedding(seq_len, n_dim)

    def forward(self, x) : 
        # x shape = (batch_size, seq_len,n_dim)
        batch_size, seq_len, n_dim=x.shape
        positions=torch.arange(seq_len, x.device, dtype=x.dtype).unsqueeze(0)
        return self.Embedding_layer(positions)
    



class GroupQueryAttention(nn.Module) : 
    def __init__(self, d_in,num_kv_groups, num_heads=None,head_dim=None,qk_norm=False,dtype=None):

        super().__init__()
        assert num_heads%num_kv_groups==0
        if head_dim==None: 
            
            assert d_in%num_heads==0 , "The dimension should be a multiple of the number of heads"
            head_dim=d_in//num_heads
        self.head_dim=head_dim
        self.num_heads=num_heads
        self.num_kv_groups=num_kv_groups
        self.group_size=num_heads//num_kv_groups
        self.d_out=num_heads*head_dim

        self.d_in=d_in
        
        self.W_query=nn.Linear(d_in , self.d_out , bias=False, dtype=dtype)
        self.W_key=nn.Linear(d_in, num_kv_groups*self.head_dim, bias=False, dtype=dtype)
        self.W_value=nn.Linear(d_in , num_kv_groups*self.head_dim, bias=False , dtype=dtype)

        self.out_proj=nn.Linear( self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm : 
            self.q_norm=RMSNorm(head_dim, eps=1e-6)
            self.k_norm=RMSNorm(head_dim, eps=1e-6)
        else : 
            self.q_norm=self.k_norm=None

    def forward (self, x,mask,cos, sin, start_pos, cache=None):
        
        batch_size, seq_len, _ =x.shape
        queries=self.W_query(x).view(batch_size,seq_len,self.num_heads, self.head_dim ).transpose(1,2)
        keys_new=self.W_key(x).view(batch_size,seq_len,self.num_kv_groups, self.head_dim ).transpose(1,2)
        values_new=self.W_value(x).view(batch_size,seq_len,self.num_kv_groups, self.head_dim ).transpose(1,2)

        # if norm 
        if self.q_norm : 
            queries=self.q_norm(queries)
        if self.k_norm :
            keys_new=self.k_norm(keys_new)

        #applys the rope on keys and queries
        queries=apply_rope(queries,cos, sin,start_pos)
        keys_new=apply_rope(keys_new, cos, sin,start_pos)

        #build group queries
        # print("logg group size", self.group_size)
        # print("logg  : n_heads", self.num_heads)
        keys_new=keys_new.repeat_interleave(self.group_size, dim=1)
        values_new=values_new.repeat_interleave(self.group_size, dim=1)

        #cache 
        if cache : 
            prev_key , prev_val=cache
            keys=torch.cat([prev_key, keys_new], dim=2)
            values=torch.cat([prev_val, values_new] , dim=2)
        else :
            start_pos=0 
            keys=keys_new
            values=values_new
        next_cache=(keys, values)
        # print("logg : queries shape",queries.shape)
        # print("logg : keys shape, ",keys.transpose(2,3).shape)
        # print("logg : value shape, ",values.shape)

        attn_scores=queries @ keys.transpose(2,3)
        attn_scores=attn_scores.masked_fill(mask , -torch.inf)
        attn_weights=torch.softmax(
            attn_scores/self.head_dim**0.5, dim=-1
        )
        context=(attn_weights @values).transpose(1,2)
        context=context.reshape(batch_size, seq_len, self.d_out)
        return self.out_proj(context) , next_cache



class FlashGroupQueryAttention(nn.Module) : 
    def __init__(self, d_in,num_kv_groups, num_heads=None,head_dim=None,qk_norm=False,dtype=None):

        super().__init__()
        assert num_heads%num_kv_groups==0
        if head_dim==None: 
            
            assert d_in%num_heads==0 , "The dimension should be a multiple of the number of heads"
            head_dim=d_in//num_heads
        self.head_dim=head_dim
        self.num_heads=num_heads
        self.num_kv_groups=num_kv_groups
        self.group_size=num_heads//num_kv_groups
        self.d_out=num_heads*head_dim

        self.d_in=d_in
        
        self.W_query=nn.Linear(d_in , self.d_out , bias=False, dtype=dtype)
        self.W_key=nn.Linear(d_in, num_kv_groups*self.head_dim, bias=False, dtype=dtype)
        self.W_value=nn.Linear(d_in , num_kv_groups*self.head_dim, bias=False , dtype=dtype)

        self.out_proj=nn.Linear( self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm : 
            self.q_norm=RMSNorm(head_dim, eps=1e-6)
            self.k_norm=RMSNorm(head_dim, eps=1e-6)
        else : 
            self.q_norm=self.k_norm=None

    def forward (self, x,mask,cos, sin, start_pos, cache=None):
        
        batch_size, seq_len, _ =x.shape
        queries=self.W_query(x).view(batch_size,seq_len,self.num_heads, self.head_dim ).transpose(1,2)
        keys_new=self.W_key(x).view(batch_size,seq_len,self.num_kv_groups, self.head_dim ).transpose(1,2)
        values_new=self.W_value(x).view(batch_size,seq_len,self.num_kv_groups, self.head_dim ).transpose(1,2)

        # if norm 
        if self.q_norm : 
            queries=self.q_norm(queries)
        if self.k_norm :
            keys_new=self.k_norm(keys_new)

        #applys the rope on keys and queries
        queries=apply_rope(queries,cos, sin,start_pos)
        keys_new=apply_rope(keys_new, cos, sin,start_pos)

        #build group queries
        # print("logg group size", self.group_size)
        # print("logg  : n_heads", self.num_heads)
        keys_new=keys_new.repeat_interleave(self.group_size, dim=1)
        values_new=values_new.repeat_interleave(self.group_size, dim=1)

        #cache 
        if cache : 
            prev_key , prev_val=cache
            keys=torch.cat([prev_key, keys_new], dim=2)
            values=torch.cat([prev_val, values_new] , dim=2)
        else :
            start_pos=0 
            keys=keys_new
            values=values_new
        next_cache=(keys, values)

        context=torch.nn.functional.scaled_dot_product_attention(
            queries,keys,values, 
            attn_mask=~mask,
            dropout_p=0.0, 
            is_causal=False,
            # enable_gqa=True, 
            scale=self.head_dim**(-0.5)
        )
        # assert torch.isfinite(context).all(), "context bad after FA"
        context=context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_out)

        return self.out_proj(context) , next_cache

