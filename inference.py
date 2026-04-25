import torch 
import yaml 
from data.dataset import create_dataloader
from model.tokenizer import Qwen3Tokenizer
from model.Qwen3 import Qwen3Model
from model.utils import generate_text_stream, calc_loss_loader , train_model_simple
def load_config_to_dict(path) : 
    with open(path, "r", encoding="utf-8") as f : 
        config=yaml.safe_load(f)

    #mappaing 
    for key, value in config['model_name'].items() : 
        if "dtype" in config['model_name'][key]:
            if config['model_name'][key]["dtype"]=="float32" : 
                config['model_name'][key]["dtype"]=torch.float32
            if config['model_name'][key]["dtype"]=="float16" : 
                config['model_name'][key]["dtype"]=torch.float16
            if config['model_name'][key]["dtype"]=="bfloat16" : 
                config['model_name'][key]["dtype"]=torch.bfloat16

    return config


config_path="F:/COURS IA/4A/PREPA INTERVIEW/AI CODING/AI/LLM projet/configs/qwen3_base_4b.yaml"
model_name="Qwen3_base_0_6B"
sample_text="how are you today?"


if  __name__=="__main__" : 
    file_path="the-verdict.txt"
    with open(file_path, 'r', encoding="utf-8") as f :
        text_data=f.read()

    train_ration=0.9
    split_idx=int(train_ration*len(text_data))
    train_data=text_data[:split_idx]
    val_data=text_data[split_idx:]



    # import the tokenizer 
    torch.manual_seed(123)
    print("logger : Run inference")
    tokenizer=Qwen3Tokenizer()
    config_model=load_config_to_dict("F:/COURS IA/4A/PREPA INTERVIEW/AI CODING/AI/LLM projet/configs/qwen3_base_4b.yaml")["model_name"]['Qwen3_base_0_6B']
    #model 
    model=Qwen3Model(config_model)

    # input_tokens=torch.tensor(tokenizer.encode(sample_text)).unsqueeze(0)
    device="cuda" if torch.cuda.is_available() else "cpu"
    # output=generate_text_stream(model,input_tokens,max_new_tokens=100,tokenizer=tokenizer,context_length=config["model_name"]['Qwen3_base_0_6B']["context_length"], device=device)

    train_loader=create_dataloader(train_data, batch_size=2, max_length=config_model['context_length'], stride=config_model["context_length"], drop_last=True, shuffle=True, num_workers=0)
    val_loader=create_dataloader(val_data,batch_size=2, max_length=config_model['context_length'] , stride=config_model['context_length'], drop_last=False, shuffle=False, num_workers=0)


    # print("train loader: ")
    # for x,y in train_loader : 
    #     print(x.shape, y.shape)


    # print("val loader")
    # for x, y, in val_loader : 
    #     print(x.shape, y.shape)

    model.to(device)
    # with torch.no_grad(): 
    #     train_loss=calc_loss_loader(train_loader, model , device)
    #     val_loss=calc_loss_loader(val_loader, model, device)

    # print('logger trainning loss: ', train_loss)
    # print("logger validation loss: ", val_loss)

    optimizer=torch.optim.AdamW(
        model.parameters(), 
        lr=0.0004, weight_decay=0.1
        )
    num_epochs=10
    train_losses , val_losses, tokens_seen=train_model_simple(
        model, train_loader, val_loader, optimizer, device, 
        num_epochs=num_epochs, eval_freq=5, eval_iter=5, 
        start_context="What is his name?", tokenizer=tokenizer, context_length=config_model["context_length"]
    )