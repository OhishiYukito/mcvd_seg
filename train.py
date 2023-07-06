# https://github.com/voletiv/mcvd-pytorch/blob/master/runners/ncsn_runner.py


from models.unet import UNet, UNet_SMLD, UNet_DDPM
from tools.functions_with_config import FuncsWithConfig
from datasets import get_dataset, data_transform, inverse_data_transform
from losses import get_optimizer
from config import dict2namespace


import torch
from torch.utils.data import DataLoader
import yaml
import argparse
from tqdm import tqdm
import wandb
import os
#import datetime


# get args
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', help="path of config (.yaml)", default='bair_01.yaml')

args = parser.parse_args()


# load config
with open('config/'+args.config_path) as f:
   dict_config = yaml.load(f, Loader=yaml.FullLoader)
config = dict2namespace(dict_config)


# Collate fn for n repeats
def my_collate(batch):
    #data = zip(*batch)
    data = torch.stack(batch).repeat_interleave(config.eval.preds_per_test, dim=0)
    #return data, torch.zeros(len(data))
    return data

# make the Dataset (Dataloader)
# https://github.com/voletiv/mcvd-pytorch/blob/master/runners/ncsn_runner.py#L254
train_dataset, test_dataset = get_dataset(config)
##### TODO make Dataset and Dataloader for Segmentation #########################################
train_dataloader = DataLoader(train_dataset, batch_size=getattr(config.train, 'batch_size', 64), shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=getattr(config.train, 'batch_size', 64)//config.eval.preds_per_test, shuffle=True, num_workers=4, drop_last=True, collate_fn=my_collate)
test_iter = iter(test_dataloader)

# make the model
model = UNet_DDPM(config)#.to(config.device)
model.train()

# function set
funcs = FuncsWithConfig(config)

# set the optimizer
optimizer = get_optimizer(config, model.parameters())
L1 = getattr(config.train, 'L1', False)

# Parallelisation
if config.device=="cuda":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
        print(f"{torch.cuda.device_count()} GPUs are avairable.")
    else:
        print(f"------- config.device is {config.device}, but GPU can't use! So we use the CPUs. -------")
        device = torch.device("cpu")
else:
    device = torch.device("cpu")


#logging_config = {}
tags = funcs.get_tags()

# training
wandb.init(
    project="MCVD with Seg",
    config=dict_config,
    name=args.config_path,
    tags=tags,
)
step = 0
for epoch in range(config.train.num_epochs):
    print(f"----------------- ↓ epoch {epoch}/{config.train.num_epochs} ---------------------")
    for i, batch in enumerate(tqdm(train_dataloader)):
        
        step += 1
        
        optimizer.zero_grad()
        
        batch = batch.to(device)
        batch = data_transform(config, batch)
        # separate frames to input(x), condition(cond)
        x, conds = funcs.separate_frames(batch)
        
        # sampling t, z, and make noisy frames
        t, z, x_t = funcs.get_noisy_frames(model, x)
        
        # make condition frames (reshaping and masking)
        masked_conds, masks = funcs.get_masked_conds(conds)      # in:(batch_size, num_frames, C, H ,W) => out:(batch_size, num_frames*C, H, W)
        
        # concat conditions (masked_past_frames + masked_future_frames)
        if masked_conds[0] is not None:
            if masked_conds[1] is not None:
                # condition = cond_frames + future_frames
                masked_conds = torch.cat(masked_conds, dim=1)
            else:
                # condition = cond_frames
                masked_conds = masked_conds[0]
        else:
            if masked_conds[1] is not None:
                # condition = future_frames
                masked_conds = masked_conds[1]
            else:
                # condition = None
                masked_conds = None
        
                
        # predict 
        predict = model(x_t, t, masked_conds)
        
        # Loss
        if L1:
            def pow_(x):
                return x.abs()
        else:
            def pow_(x):
                return 1 / 2. * x.square()
        loss = pow_((z - predict).reshape(len(x), -1)).sum(dim=-1)
        loss = loss.mean(dim=0) 
        
        loss.backward()
        optimizer.step()

        # Log
        if config.train.logging and i % config.train.log_interval == 0:
            if config.train.validation:
                # Validation
                with torch.no_grad():
                    try:
                        test_batch = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_dataloader)
                        test_batch = next(test_iter)
                    
                    test_batch = data_transform(config, test_batch)
                    
                    target, conds_test = funcs.separate_frames(test_batch.to(device))
                    target = target.reshape(target.shape[0], -1, target.shape[-2], target.shape[-1])
                    
                    accuracies = {}
                    
                    for task in tags:
                        if task == "generation":
                            prob_mask_p = 1.0
                            prob_mask_f = 1.0
                        elif task == "interpolation":
                            prob_mask_p = 0.0
                            prob_mask_f = 0.0
                        elif task == "past_prediction":
                            prob_mask_p = 1.0
                            prob_mask_f = 0.0
                        elif task == "future_prediction":
                            prob_mask_p = 0.0
                            prob_mask_f = 1.0
                        
                        masked_conds_test, masks = funcs.get_masked_conds(conds_test, prob_mask_p, prob_mask_f)
                        
                        # concat conditions (masked_past_frames + masked_future_frames)
                        if masked_conds_test[0] is not None:
                            if masked_conds_test[1] is not None:
                                # condition = cond_frames + future_frames
                                masked_conds_test = torch.cat(masked_conds_test, dim=1)
                            else:
                                # condition = cond_frames
                                masked_conds_test = masked_conds_test[0]
                        else:
                            if masked_conds_test[1] is not None:
                                # condition = future_frames
                                masked_conds_test = masked_conds_test[1]
                            else:
                                # condition = None
                                masked_conds_test = None
                                
                        init_batch = funcs.get_init_sample(model, target.shape)
                        # Get predicted x_0
                        pred = funcs.reverse_process(model, init_batch, masked_conds_test, final_only=True)  # pred : ['0' if final_only else 'len(steps)', B, C*F, H, W]
                        pred = inverse_data_transform(config, pred[-1])
                        
                        # Calculate accuracy with target, pred
                        target = inverse_data_transform(config, target)
                        conds_test = [inverse_data_transform(config, d) if d is not None else None for d in conds_test]
                        accuracies = funcs.get_accuracy(pred, target, conds_test)
                        # TODO save accuracies
                
                                
            # logging with wandb
            wandb.log(data={"loss": loss, "step[epoch]": epoch+(i+1)/len(train_dataloader)},
                      step= step
                      )
            
            
            # Save model
            states = [model.state_dict(),
                      optimizer.state_dict(),
                      epoch,
                      step
                      ]
            folder_path = os.path.join('results', config.data.dataset.upper(), args.config_path.replace(".yaml", ""))
            os.makedirs(folder_path, exist_ok=True)
            #dt = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
            #ckpt_path = os.path.join(folder_path, '-'.join(tags)+'__'+dt+'.pt')  # 'results/[DATASET]/[CONFIG]/[TASK1]-[TASK2]__[DATETIME].pt'
            ckpt_path = os.path.join(folder_path, '-'.join(tags)+'.pt')          # 'results/[DATASET]/[CONFIG]/[TASK1]-[TASK2].pt'
            torch.save(states, ckpt_path)
print("finish train.py!")