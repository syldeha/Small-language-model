import torch 
from torch.utils.data import Dataset , DataLoader, IterableDataset
from model.tokenizer import Qwen3Tokenizer
import numpy as np



class MemTokenDataset(IterableDataset) : 
    def __init__(self, bin_path, context_length, batch_size): 
        self.bin_path=bin_path
        self.context_length=context_length
        self.batch_size=batch_size


    def __iter__(self): 
        data=np.memmap(self.bin_path, dtype=np.uint16, mode='r')
        max_start=len(data)-self.context_length-1

        while True: 
            ix=torch.randint(0,max_start, (self.batch_size,))  #select (batch_size) indexes start randomly from (1 to max_start)

            x=torch.stack([
                torch.from_numpy(np.array(data[i:i+self.context_length],dtype=np.int64 )) for i in ix
            ])

            y=torch.stack([
                torch.from_numpy(np.array(data[i+1:i+1+self.context_length] , dtype=np.int64)) for i in ix
            ])

            yield x ,y






class QwenDataset(Dataset) : 
    def __init__(self, txt, tokenizer, max_length, stride) : 
        """
        txt : [] List of string 
        """
        self.input_ids=[]
        self.target_ids=[]


        token_ids_list=tokenizer.encode(txt) #we tokenize the entire text adn return a list of ids
        for i in range(0 , len(token_ids_list)-max_length, stride) : 
            input_chunk=token_ids_list[i:i+max_length]
            target_chunks=token_ids_list[i+1:i+1+max_length]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunks))


    def __len__(self):
        return len(self.input_ids)
        

    def __getitem__(self, idx):
        return self.input_ids[idx] , self.target_ids[idx]
    
class QwenDataset_(Dataset) : 
    def __init__(self, path_txt, tokenizer, max_length, stride) : 
        """
        txt : [] List of string 
        """
        self.input_ids=[]
        self.target_ids=[]
        file_path="the-verdict.txt"
        with open(file_path, 'r', encoding="utf-8") as f :
            text_data=f.read()
        self.txt=text_data

        token_ids_list=tokenizer.encode(self.txt) #we tokenize the entire text adn return a list of ids
        for i in range(0 , len(token_ids_list)-max_length, stride) : 
            input_chunk=token_ids_list[i:i+max_length]
            target_chunks=token_ids_list[i+1:i+1+max_length]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunks))


    def __len__(self):
        return len(self.input_ids)
        

    def __getitem__(self, idx):
        return self.input_ids[idx] , self.target_ids[idx]
    




def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=1) : 
    """
    txt : [list of string]
    drop_last : Drops or not the last batch if it is shorter than the specified batch size
    num_workers : The number of CPU processes to use for peprocessing

    """
    tokenizer=Qwen3Tokenizer()
    dataset=QwenDataset(txt, tokenizer, max_length, stride)
    dataloader=DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle, 
        drop_last=drop_last , 
        num_workers=num_workers 

    )


    return dataloader