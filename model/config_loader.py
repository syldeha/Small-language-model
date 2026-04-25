# config_loader.py
import yaml
import os
import random
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str) -> DictConfig:
    """
    Load cfg from YAML using OmegaConf.
    
    Args:
        config_path: Path to YAML cfg file
        
    Returns:
        cfg: OmegaConf DictConfig object
    """
    # Load with OmegaConf
    cfg = OmegaConf.load(config_path)
    
    # 1. ENVIRONMENT VARIABLES
    if "setting" in cfg and "os_environ" in cfg.setting:
        for key, value in cfg.setting.os_environ.items():
            if value is not None:
                os.environ[key] = str(value)
                
    
    # 2. SET RANDOM SEED
    if "setting" in cfg and "seed" in cfg.setting:
        seed = cfg.setting.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    
    # 3. VALIDATE REQUIRED FIELDS
    required_sections = ["model", "training", "dataset", "Trainer"]
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing required section: {section}")

    required_model_keys = ["vocab_size", "emb_dim", "n_heads", "n_layers", "context_length"]
    for key in required_model_keys:
        if key not in cfg.model:
            raise ValueError(f"Missing required model cfg: {key}")
    
    # plain dicts:
    # cfg = OmegaConf.to_container(cfg)
  
    # print(OmegaConf.to_yaml(cfg)) 
    
    return cfg


# USAGE:
# if __name__ == "__main__":
#     cfg = load_config("./configs/training_config.yaml")
    
#     # Much cleaner access with dot notation:
#     print(cfg.model.emb_dim)                    # 1024 (no quotes!)
#     print(cfg.training.optimizer_kwargs.type)   # AdamW
#     print(cfg.Trainer.devices)                  # 1
    
#     print(cfg["model"]["emb_dim"])              # Still works
    
#     # Powerful features:
#     print(OmegaConf.to_yaml(cfg))              # Print entire cfg nicely
#     # OmegaConf.to_yaml(cfg, "output.yaml")      # Save modified cfg