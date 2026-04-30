import lightning as L
import torch 
import inspect
from model.Qwen3 import Qwen3Model
from model.tokenizer import Qwen3Tokenizer
from model.utils import calc_loss_batch, generate_and_print_sample

from model.muon import Muon
from omegaconf import OmegaConf
DTYPE_MAP={
    "float32": torch.float32, 
    "float16": torch.float16, 
    "bfloat16" : torch.bfloat16
}

class QwenModel(L.LightningModule) : 
    def __init__(self,config) : 
        super().__init__()
        self.config=config
        model_config=OmegaConf.to_container(config.model, resolve=True)
        model_config["dtype"]=DTYPE_MAP[model_config["dtype"]]
        self.model=Qwen3Model(model_config)
        self.token_seen=0
        self.tokenizer=Qwen3Tokenizer()
        self.start_context=config.training.generate_sample.start_context
        self.context_length=config.model.context_length
        self.save_hyperparameters()
        self.automatic_optimization=False


    def forward(self, inputs): 
        return self.model(inputs)


    def training_step(self,batch, batch_idx) : 
        optim_adamw, optim_muon=self.optimizers()
        inputs, target= batch
        device=inputs.device

        loss=calc_loss_batch(
            inputs, target, self.model, device
        )
        local_tokens=torch.tensor(inputs.numel(), device=self.device)
        global_tokens=self.all_gather(local_tokens).sum()

        self.token_seen+=int(global_tokens.item())

        self.manual_backward(loss)
        optim_adamw.step()
        optim_muon.step()

        optim_adamw.zero_grad(set_to_none=True)
        optim_muon.zero_grad(set_to_none=True)

        self.log("token_seen" , float(self.token_seen), on_step=True, prog_bar=True, logger=True)
        self.log("train_loss" , loss, on_step=True, on_epoch=True,  prog_bar=True, logger=True)

        return loss

    def validation_step(self,batch, batch_idx):
        inputs, target= batch
        loss=calc_loss_batch(
            inputs, target, self.model, inputs.device
        )

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        if not self.trainer.is_global_zero:
            return
        
        generate_and_print_sample(
            self.model, self.tokenizer, self.device, self.start_context, context_length=self.context_length
        )

    # def configure_optimizers(self) : 
    #     return torch.optim.AdamW(
    #         self.parameters(),
    #         lr=self.config.training.lr_scheduler_kwargs.init_lr, 
    #         betas=tuple(self.config.training.optimizer_kwargs.betas),
    #         weight_decay=self.config.training.optimizer_kwargs.weight_decay,
    #     )


    # def configure_optimizers(self):  #nanochat
    #     weight_decay=self.config.training.optimizer_kwargs.weight_decay
    #     learning_rate=self.config.training.lr_scheduler_kwargs.init_lr
    #     betas=tuple(self.config.training.optimizer_kwargs.betas)
    #     # start with all of the candidate parameters
    #     param_dict = {pn: p for pn, p in self.named_parameters()}
    #     # filter out those that do not require grad
    #     param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    #     # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    #     # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.

    #     decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    #     nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    #     optim_groups = [
    #         {'params': decay_params, 'weight_decay': weight_decay},
    #         {'params': nodecay_params, 'weight_decay': 0.0}
    #     ]
    #     num_decay_params = sum(p.numel() for p in decay_params)
    #     num_nodecay_params = sum(p.numel() for p in nodecay_params)
    #     print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    #     print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    #     # Create AdamW optimizer and use the fused version if it is available
    #     fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    #     use_fused = fused_available and self.device == 'cuda'
    #     extra_args = dict(fused=True) if use_fused else dict()
    #     optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    #     print(f"using fused AdamW: {use_fused}")

        # return optimizer
    def configure_optimizers(self) : 
        adamw_config=self.config.training.optimizer_kwargs.adamw
        muon_config=self.config.training.optimizer_kwargs.muon
        param_dict={ pn:p for pn,p in self.named_parameters()}
        param_dict={pn:p for pn,p in param_dict.items() if p.requires_grad}
        muons_params=[]
        adamw_params=[]

        for pn,p in param_dict.items() : 
            is_hidden_params=(
                p.ndim<=2 
                and "scale" not in pn 
                and "tok_emb" not in pn
                and "out_head" not in pn
            )
            if is_hidden_params : 
                muons_params.append(p)
            else : 
                adamw_params.append(p)
        optimizers_muon=Muon(
            muons_params, 
            lr=muon_config.lr, 
            weight_decay=muon_config.weight_decay,
            momentum=muon_config.momentum, 
            nesterov=muon_config.nesterov,
            ns_steps=muon_config.ns_steps, 

        )

        optimizers_adamw=torch.optim.AdamW(
            adamw_params , 
            lr=adamw_config.lr, 
            betas=adamw_config.betas, 
            weight_decay=adamw_config.weight_decay
        )
        return  [optimizers_adamw, optimizers_muon]
        
    

    
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
#     batch=next(iter(dm.train_dataloader()))

#     model=QwenModel(config)

#     inputs , target= batch
#     loss=calc_loss_batch(inputs, target, model.model, inputs.device)
#     print("logger : ", loss)

#     #next test 
#     trainder=L.Trainer(
#         fast_dev_run=1, 
#         accelerator="gpu",
#         devices=1, 
#         logger=True, 
#         enable_checkpointing=False, 


#     )
#     trainder.fit(model, datamodule=dm)

    
