import torch
from tqdm import tqdm

class KVCache : 
    def __init__(self, n_layers) : 
        self.cache=[None]*n_layers


    def get(self, layer_idx) : 
        return self.cache[layer_idx]
    
    def update(self, layer_idx, value) : 
        self.cache[layer_idx]=value

    def get_all(self) : 
        return self.cache
    
    def reset(self) : 
        for i in range(len(self.cache)) : 
            self.cache[i]=None


def generate_text_simple(model, idx, max_new_tokens, context_length): 
    """
        idx : idx is a (batch , n_tokens) arrays of indices in the current context
        context_size : int , max context size that the model can handle
    """
    for _ in range (max_new_tokens) : 
        idx_cond=idx[: , -context_length: ] #cropped the size of the input to keep the size that the model can handle
        with torch.no_grad():
            logits=model(idx_cond)

        logits=logits[:,-1,:] #the output size is (batch, n_tokens, vocab) and we keep only the last one which is the next token predicted
        probas=torch.softmax(logits, dim=-1) # probas has the shape (batch , vocab_size)
        idx_next=torch.argmax(probas, dim=-1, keepdim=True) #(batch,1)
        idx=torch.cat([idx, idx_next], dim=1) #appends sampled index to the running sequence and the new one has the shape (batch ,n_tokens+1)

    return idx



import time
device="cuda" if torch.cuda.is_available() else "cpu"
def generate_text_stream (model ,idx, max_new_tokens,tokenizer, context_length,device) : 
    """
    idx : (n_batch , n_tokens)

    """
    model.eval()
    model.to(device)
    idx=idx.to(device)
    generate_count=0
    start_time=time.time()
    print("Input_text: \n ")
    print(tokenizer.decode(idx[0].tolist()), end="")
    print("\n ")
    for _ in range(max_new_tokens) : 
        idx_cond=idx[:, -context_length:]
        logits=model(idx_cond)[:,-1,:]
        probas=torch.softmax(logits, dim=-1) #(batch, vocab_size)
        idx_new=torch.argmax(probas, dim=-1, keepdim=True) #(batch, 1)

        generate_count+=1
        idx=torch.cat([idx, idx_new], dim=1)

        #print
        print(tokenizer.decode(idx_new[0].tolist()), end="" , flush=True)

    elapsed=time.time()-start_time
    tok_per_sec=generate_count/elapsed if elapsed> 0 else None

    print('\n')
    print(tokenizer.decode(idx[0].tolist()))
    print(f"\n time {elapsed:.2f} sec")
    print(f" Tokens/sec {tok_per_sec:.2f}")


    return idx

def generate_text_stream_top_k (model ,idx, max_new_tokens,tokenizer, context_length,device, top_k=5, temperature=0.5) : 
    """
    idx : (n_batch , n_tokens)

    """
    model.eval()
    model.to(device)
    idx=idx.to(device)
    generate_count=0
    start_time=time.time()
    print("Input_text: \n ")
    print(tokenizer.decode(idx[0].tolist()), end="")
    print("\n ")
    for _ in range(max_new_tokens) : 
        idx_cond=idx[:, -context_length:]
        logits=model(idx_cond)[:,-1,:]
        if top_k is not None: 
            top_logits, _=torch.topk(logits, top_k)
            min_val=top_logits[:,-1]
            logits=torch.where(
                logits< min_val , 
                # torch.tensor(float('-inf')).to(logits.device),  recreates the tensor every steps better use torch.full_likes
                torch.full_like(logits, float("-inf")),
                logits
            )
        if temperature > 0.0 : 
            logits=logits/temperature
            probs=torch.softmax(logits, dim=-1)
            idx_new=torch.multinomial(probs, num_samples=1)
        else : 
            idx_new=torch.argmax(logits, dim=-1, keepdim=True)

        if idx_new.item()== tokenizer.eos_token_id : 
            break
        generate_count+=1
        idx=torch.cat([idx, idx_new], dim=1)

        #print
        print(tokenizer.decode(idx_new[0].tolist()), end="" , flush=True)

    elapsed=time.time()-start_time
    tok_per_sec=generate_count/elapsed if elapsed> 0 else None

    print('\n')
    print(tokenizer.decode(idx[0].tolist()))
    print(f"\n time {elapsed:.2f} sec")
    print(f" Tokens/sec {tok_per_sec:.2f}")


    return idx
        

def text_to_token_ids(text, tokenizer): 
    """
    text : [] list of 
    """
    encoded=tokenizer.encode(text)
    encoded_tensor=torch.tensor(encoded).unsqueeze(0)

    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer) : 
    flat=token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())
    
def calc_loss_batch(input_batch, target_batch, model, device) : 
    input_batch=input_batch.to(device)
    target_batch=target_batch.to(device)
    logits=model(input_batch)
    # print(f"Output: min={logits.min()}, max={logits.max()}, has_nan={torch.isnan(logits).any()}")
    # logits=logits.float()
    # logits=15.0*torch.tanh(logits/15.0)
    loss=torch.nn.functional.cross_entropy(
        logits.flatten(0,1) , target_batch.flatten()
    )
    """
    flatten(0,1) flat the first two dimensional input (batch, n_token, vocab_size) ---> (batch*n_token, vocab_size)
    target_batch (n_batch, n_tokens) ---> flatten  (n_batch*n_tokens)
    So we have  : logits.flatten(0,1) contains "n_batch*n_tokens" vectors of size vocab_size
                 * target_batch.flatten() contains "n_batch*n_tokens" integers, coresponding to the index of the target we will use to compute the cross entropy

    
    """

    return loss


#function to cmmpute the training and the validation loss
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss=0.0
    if len(data_loader)==0 : 
        return float('nan')
    
    elif num_batches is None : 
        num_batches=len(data_loader)
    else: 
        num_batches=min(num_batches, len(data_loader))
    for i , (input_batch, target_batch) in enumerate(data_loader) : 
        if i<num_batches :
            loss=calc_loss_batch(
                input_batch=input_batch, 
                target_batch=target_batch, 
                model=model, device=device
            )
            total_loss+=loss.item()

        else : 
            break
    return total_loss/num_batches
 



# print the 
def evaluate_model(model, train_loader, val_loader, device, eval_iter) : 
    model.eval()
    with torch.no_grad() : 
        train_loss=calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss=calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )

    model.train()

    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context, context_length):
    model.eval()


    encoded=text_to_token_ids(start_context,tokenizer).to(device)

    with torch.no_grad():
        token_ids=generate_text_stream_top_k(
            model=model, 
            idx=encoded,
            max_new_tokens=50, 
            context_length=context_length,
            tokenizer=tokenizer,
            device=device
                      )
        decoded_text=token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n"," "))

        model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer, context_length) : 
    train_losses, val_losses, track_tokens_seen= [] , [] , []
    """
    eval_iter : represent the number of batches used during the evaluation
    """

    token_seen , global_step =0 , -1
    for epoch in range(num_epochs) : 
        model.train()

        for input_batch , target_batch in tqdm(train_loader):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss=calc_loss_batch(
                    input_batch,target_batch,model,device
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters() , 1.0)

            optimizer.step()
            token_seen+=input_batch.numel()
            global_step+=1

            if global_step % eval_freq==0 : 
                train_loss, val_loss=evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                track_tokens_seen.append(token_seen)
                print(f"EP {epoch+1} (Step {global_step:06d}) : "
                      f"Train loss {train_loss: .3f} ,"
                      f"Val loss {val_loss: .3f}"
                      
                      )
        
        generate_and_print_sample(
            model, tokenizer, device, start_context,context_length=context_length
        )

    return train_losses, val_losses, track_tokens_seen

