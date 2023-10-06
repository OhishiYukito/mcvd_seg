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
import time
import numpy as np
import gc
import sys

# get args
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', help="path of config (.yaml)", default='kth64_01_deeper_5_5_10.yaml')

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
train_dataset, test_dataset = get_dataset(config, segmentation=False)

train_dataloader = DataLoader(train_dataset, batch_size=getattr(config.train, 'batch_size', 64), shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=getattr(config.train, 'batch_size', 64)//config.eval.preds_per_test, shuffle=True, num_workers=4, drop_last=True, collate_fn=my_collate)
test_iter = iter(test_dataloader)
# make Dataset and Dataloader for Segmentation
if 0.0<config.model.prob_mask_s<1.0:
    seg_train_dataset, seg_test_dataset = get_dataset(config, segmentation=True)
    seg_train_dataloader = DataLoader(seg_train_dataset, batch_size=getattr(config.train, 'batch_size', 64), shuffle=True, num_workers=100)
    def seg_test_collate(batch):
        origin_batch = torch.stack([data[0] for data in batch]).repeat_interleave(config.eval.preds_per_test, dim=0)
        ann_batch = torch.stack([data[1] for data in batch]).repeat_interleave(config.eval.preds_per_test, dim=0)
        return origin_batch, ann_batch
    seg_test_dataloader = DataLoader(seg_test_dataset, batch_size=getattr(config.train, 'batch_size', 64)//config.eval.preds_per_test, shuffle=True, num_workers=4,
                                     drop_last=True, collate_fn=seg_test_collate)

# make the model
model = UNet_DDPM(config)#.to(config.device)
model.train()

# function set
funcs = FuncsWithConfig(config)
if 0.0<config.model.prob_mask_s<1.0:
    funcs.seg_train_dataloader = seg_train_dataloader
    funcs.seg_test_dataloader = seg_test_dataloader
    funcs.seg_train_iter = iter(seg_train_dataloader)
    funcs.seg_test_iter = iter(seg_test_dataloader)

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
if not __debug__:
    wandb.init(
        project="MCVD with Seg",
        config=dict_config,
        name=args.config_path,
        tags=tags,
    )
step = 0
high_loss = False
for epoch in range(config.train.num_epochs):
    print(f"----------------- ↓ epoch {epoch}/{config.train.num_epochs} ---------------------")
    #if epoch==1:
    #    break
    for i, batch in enumerate(tqdm(train_dataloader)):
    #    start_loop = time.time()
    #    if i==3:
    #        break
        step += 1
        
        optimizer.zero_grad()
        
        batch = batch.to(device)
        batch = data_transform(config, batch)
        # separate frames to input(x), condition(cond)
        x, conds = funcs.separate_frames(batch)

        # make condition frames (reshaping and masking)
        #start = time.time()
        masked_conds, masks = funcs.get_masked_conds(conds)      # in:(batch_size, num_frames, C, H ,W) => out:(batch_size, num_frames*C, H, W)
        #print("get_masked_conds(): {} [s]".format(time.time()-start))
        del conds, batch

        # concat conditions (masked_past_frames + masked_future_frames + masked_seg_frames)
        if masked_conds[0] is not None:
            if masked_conds[1] is not None:
                # condition = cond_frames + future_frames
                masked_conds_train = torch.cat(masked_conds[:2], dim=1)
            else:
                # condition = cond_frames
                masked_conds_train = masked_conds[0]
        else:
            if masked_conds[1] is not None:
                # condition = future_frames
                masked_conds_train = masked_conds[1]
            else:
                # condition = None
                masked_conds_train = None
        
        if masked_conds[2][0] is not None:
            if masked_conds_train is not None:
                # condition += seg_frames
                masked_conds_train = torch.cat([masked_conds_train, masked_conds[2][0]], dim=1)
            else:
                # condition = seg_frames
                masked_conds_train = masked_conds[2][0]
                
            # change input frames for segmentation
            if masks[2] is not None:
                for index in range(len(x)):
                    if masks[2][index]==True:
                        # replace input frames to 'segmentation' from 'frame_generation'
                        x[index] = masked_conds[2][1][index].reshape(config.data.num_frames, -1, x[index].shape[-2], x[index].shape[-1])   # seg_annotaion
        del masks, masked_conds
        
        # sampling t, z, and make noisy frames    
        #start = time.time()
        t, z, x_t = funcs.get_noisy_frames(model, x)
        #print("get_noisy_frames(): {} [s]".format(time.time() - start))

        # predict
        #start = time.time()
        predict = model(x_t, t, masked_conds_train)
        #print("predict: {} [s]".format(time.time()-start))
        del t, x_t
        # Loss
        #start = time.time()
        if L1:
            def pow_(x):
                return x.abs()
        else:
            def pow_(x):
                return 1 / 2. * x.square()
        loss = pow_((z - predict).reshape(len(x), -1)).sum(dim=-1)
        loss = loss.mean(dim=0) 
        del x, z, predict
        gc.collect()

        if epoch>10 and loss > 10000 and high_loss==False:
            print("loss over 10,000 at {} steps in epoch {}!!".format(i, epoch))
            high_loss = (i, epoch)
            #sys.exit()
        loss.backward()
        optimizer.step()
        ## TODO #################################################
        # if loss skyrockets, check the optimizer's parameters to confirm whether hyperparameters are correct or not.
        #########################################################
        #print("get loss ~ backward: {} [s]".format(time.time()-start))

        # Log
        if config.train.logging and ((i+1)%config.train.log_interval==0 or i==len(train_dataloader)-1):
            if config.train.validation:
                # Validation for only interpolation
                with torch.no_grad():
                    try:
                        test_batch = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_dataloader)
                        test_batch = next(test_iter)
                    
                    test_batch = data_transform(config, test_batch)
                    
                    target, conds_test = funcs.separate_frames(test_batch.to(device))       # (B, F, C, H, W)
                    target = target.reshape(target.shape[0], -1, target.shape[-2], target.shape[-1])    # (B, F*C, H, W)
                    
                    # interpolation
                    masked_conds, _ = funcs.get_masked_conds(conds_test, prob_mask_p=0.0, prob_mask_f=0.0, prob_mask_s=1.0, mode='test')
                    
                    # concat conditions (masked_past_frames + masked_future_frames)
                    if masked_conds[0] is not None:
                        if masked_conds[1] is not None:
                            # condition = cond_frames + future_frames
                            masked_conds_test = torch.cat(masked_conds[:2], dim=1)
                        else:
                            # condition = cond_frames
                            masked_conds_test = masked_conds[0]
                    else:
                        if masked_conds[1] is not None:
                            # condition = future_frames
                            masked_conds_test = masked_conds[1]
                        else:
                            # condition = None
                            masked_conds_test = None
                    
                    if masked_conds[2][0] is not None:
                        if masked_conds_test is not None:
                            # condition += seg_frames
                            masked_conds_test = torch.cat([masked_conds_test, masked_conds[2][0]], dim=1)
                        else:
                            # condition = seg_frames
                            masked_conds_test = masked_conds[2][0]
                            
                        ## change input frames for segmentation
                        #for index in range(len(target)):
                        #    if masks[2][index]==True:
                        #        # replace input frames to 'segmentation' from 'frame_generation'
                        #        target[index] = masked_conds[2][1][index].reshape(config.data.num_frames, -1, target[index].shape[-2], target[index].shape[-1])   # seg_annotaion
                            
                    init_batch = funcs.get_init_sample(model, target.shape)
                    # Get predicted x_0
                    pred = funcs.reverse_process(model, init_batch, masked_conds_test, subsample_steps=config.eval.subsample_steps, final_only=True)  # pred : ['0' if final_only else 'len(steps)', B, C*F, H, W]
                    pred = inverse_data_transform(config, pred[-1])
                    #if task == "segmentation":
                    #    # convert to 0 or 1
                    #    pred = pred > 0.5
                    #    pred = pred.to(torch.int32)
                    
                    # Calculate accuracy with target, pred
                    target = inverse_data_transform(config, target)
                    conds_test = [inverse_data_transform(config, d) if d is not None else None for d in conds_test]
                    accuracies = funcs.get_accuracy(pred, target, conds_test, 
                                                    calc_fvd=False, only_embedding=False)
                    del init_batch, pred, target, conds_test,
                    
                    calc_result = {}
                    def get_avg_std_from_best_score_list(best_score_list):
                        avg, std = best_score_list.mean().item(), best_score_list.std().item()
                        return round(avg,3), round(std,3)

                    for key in accuracies.keys():
                        if key in ["mse", "ssim", "lpips"]:
                            score_list = np.array(accuracies[key]).reshape((-1, config.eval.preds_per_test))
                            # get best score list
                            if key=="mse":
                                score_list = score_list.min(-1)
                                # calc and save psnr_score
                                psnr_list = (10 * np.log10(1 / np.array(accuracies[key]))).reshape((-1, config.eval.preds_per_test)).max(-1)
                                avg, std = get_avg_std_from_best_score_list(psnr_list)
                                calc_result["psnr"] = {"avg": avg, "std": std}
                            elif key=="ssim":
                                score_list = score_list.max(-1)
                            elif key=="lpips":
                                score_list = score_list.min(-1)
                            # get avg, std
                            avg, std = get_avg_std_from_best_score_list(score_list)
                            calc_result[key] = {"avg": avg, "std": std}
                            
                    if not __debug__:     
                        wandb.log(data={"interp_mse": calc_result["mse"]["avg"],
                                        "interp_psnr": calc_result["psnr"]["avg"],
                                        "interp_ssim": calc_result["ssim"]["avg"],
                                        "interp_lpips": calc_result["lpips"]["avg"]},
                                    step= step,
                                    commit=False)
                                
                
            if not __debug__:                    
                # logging with wandb
                wandb.log(data={"loss": loss, "step[epoch]": epoch+(i+1)/len(train_dataloader)},
                        step= step, commit=True
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
        
        #print("one train_step: {} [s]".format(time.time()-start_loop))

if high_loss != False:
    print("loss over 10,000 at {} steps in epoch {}!!".format(high_loss[0], high_loss[1]))
print("finish train.py!")
