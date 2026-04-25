from model.config_loader import load_config 
from model.utils import calc_loss_batch
from model.qwen3Module import QwenModel
from data.datamodule import QwenDataModule
import  lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

config=load_config("configs/training_config.yaml")

# dm=QwenDataModule(
#     text_data_path=config.dataset.dataset_by_path, 
#     max_length=config.dataset.dataloader_kwargs.max_length, 
#     stride=config.dataset.dataloader_kwargs.stride,
#     batch_size=config.dataset.dataloader_kwargs.batch_size,
#     num_workers=config.dataset.dataloader_kwargs.num_workers,
#     seed=config.setting.seed
# )
dm = QwenDataModule(
    train_bin_path=config.dataset.train_bin_path,
    val_bin_path=config.dataset.val_bin_path,
    context_length=config.dataset.dataloader_kwargs.max_length,
    batch_size=config.dataset.dataloader_kwargs.batch_size,
    num_workers=config.dataset.dataloader_kwargs.num_workers,
)


model=QwenModel(config)

#checkpoint 

checkpoint_callback = ModelCheckpoint(
    dirpath=config.Trainer.default_root_dir,
    monitor=config.Trainer.checkpoint.monitor,
    save_top_k=config.Trainer.checkpoint.save_top_k,
    mode=config.Trainer.checkpoint.mode,
    save_last=config.Trainer.checkpoint.save_last,
)
early_stopping_callback = EarlyStopping(
    monitor=config.Trainer.callbacks.early_stopping.monitor,
    patience=config.Trainer.callbacks.early_stopping.patience,
    mode=config.Trainer.callbacks.early_stopping.mode,
)


#next test 
trainer = L.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=config.Trainer.devices,
    max_steps=config.Trainer.max_steps,
    limit_val_batches=config.Trainer.limit_val_batches,
    log_every_n_steps=config.Trainer.log_every_n_steps,
    val_check_interval=config.Trainer.val_check_interval,
    logger=config.Trainer.logger,
    enable_checkpointing=config.Trainer.enable_checkpointing,
    default_root_dir=config.Trainer.default_root_dir,
    accumulate_grad_batches=config.Trainer.accumulate_grad_batches,
    precision=config.Trainer.precision,
    callbacks=[checkpoint_callback, early_stopping_callback],
)

# trainer.fit(model, datamodule=dm, ckpt_path=config.training.load_checkpoint)
trainer.fit(model, datamodule=dm)
