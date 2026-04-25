import lightning as L
from model.tokenizer import Qwen3Tokenizer
from data.dataset import QwenDataset
from torch.utils.data import random_split, DataLoader
import torch
from model.config_loader import load_config

# class QwenDataModule(L.LightningDataModule) : 
#     def __init__(self, text_data_path, max_length, stride, batch_size, num_workers, seed, train_ratio=0.9):
#         super().__init__()
#         with open(text_data_path , "r" , encoding="utf-8") as f: 
#             self.text_data=f.read()
#         self.max_length=max_length
#         self.stride=stride
#         self.batch_size=batch_size
#         self.num_workers=num_workers
#         self.seed=seed
#         self.train_ratio=train_ratio

#     def setup(self, stage=None)  : 
#         tokenizer=Qwen3Tokenizer()
#         full_dataset=QwenDataset(
#             txt=self.text_data, 
#             tokenizer=tokenizer,
#             max_length=self.max_length ,
#             stride=self.stride
#         ) 
#         train_size=int(self.train_ratio*len(full_dataset))
#         val_size=len(full_dataset)-train_size
#         generator=torch.Generator().manual_seed(self.seed)

#         self.train_dataset, self.val_dataset=random_split(
#             full_dataset, [train_size, val_size], 
#             generator=generator
#         )


#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset, 
#             batch_size=self.batch_size , 
#             shuffle=True, 
#             num_workers=self.num_workers
#         )

#     def val_dataloader(self) : 
#         return DataLoader(
#             self.val_dataset, 
#             batch_size=self.batch_size, 
#             shuffle=False, 
#             num_workers=self.num_workers
#         )

import lightning as L
from torch.utils.data import DataLoader
from data.dataset import MemTokenDataset


class QwenDataModule(L.LightningDataModule):
    def __init__(self, train_bin_path, val_bin_path, context_length, batch_size, num_workers=0):
        super().__init__()
        self.train_bin_path = train_bin_path
        self.val_bin_path = val_bin_path
        self.context_length = context_length
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = MemTokenDataset(
            bin_path=self.train_bin_path,
            context_length=self.context_length,
            batch_size=self.batch_size,
        )
        self.val_dataset = MemTokenDataset(
            bin_path=self.val_bin_path,
            context_length=self.context_length,
            batch_size=self.batch_size,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
        )

# if __name__=="__main__" : 
#     config = load_config("./configs/training_config.yaml")
#     dm=QwenDataModule(
#         text_data_path=config.dataset.dataset_by_path, 
#         max_length=config.dataset.dataloader_kwargs.max_length, 
#         stride=config.dataset.dataloader_kwargs.stride,
#         batch_size=config.dataset.dataloader_kwargs.batch_size,
#         num_workers=config.dataset.dataloader_kwargs.num_workers,
#         seed=config.setting.seed
#     )
#     dm.setup("fit")
#     print("logger : ", len(dm.train_dataset))
#     print("logger : ", len(dm.val_dataset))
    