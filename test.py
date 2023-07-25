from random import sample
from models.unet import UNet, UNet_SMLD, UNet_DDPM
from tools.functions_with_config import FuncsWithConfig
from datasets import get_dataset
from config import dict2namespace
from datasets import get_dataset, data_transform, inverse_data_transform
from models.fvd.fvd import frechet_distance, load_i3d_pretrained
import models.eval_models as eval_models

import torch
from torch.utils.data import DataLoader
import yaml
import os
import numpy as np
import pickle
import argparse
from tqdm import tqdm
import copy


# get args
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', help="path of config (.yaml)", default='bair_04.yaml')

args = parser.parse_args()


# load config
with open('config/'+args.config_path) as f:
   dict_config = yaml.load(f, Loader=yaml.FullLoader)
config = dict2namespace(dict_config)


def my_collate(batch):
    #data = zip(*batch)
    data = torch.stack(batch).repeat_interleave(config.eval.preds_per_test, dim=0)
    #return data, torch.zeros(len(data))
    return data

# make the Dataset (Dataloader)
# https://github.com/voletiv/mcvd-pytorch/blob/master/runners/ncsn_runner.py#L254
_, test_dataset = get_dataset(config)
test_dataloader = DataLoader(test_dataset, batch_size=getattr(config.train, 'batch_size', 64)//config.eval.preds_per_test, shuffle=True, num_workers=4, drop_last=True, collate_fn=my_collate)
if 0.0<config.model.prob_mask_s<1.0:
    seg_train_dataset, seg_test_dataset = get_dataset(config, segmentation=True)
    seg_train_dataloader = DataLoader(seg_train_dataset, batch_size=getattr(config.train, 'batch_size', 64), shuffle=True, num_workers=100)
    def seg_test_collate(batch):
        origin_batch = torch.stack([data[0] for data in batch]).repeat_interleave(config.eval.preds_per_test, dim=0)
        ann_batch = torch.stack([data[1] for data in batch]).repeat_interleave(config.eval.preds_per_test, dim=0)
        return origin_batch, ann_batch
    seg_test_dataloader = DataLoader(seg_test_dataset, batch_size=getattr(config.train, 'batch_size', 64)//config.eval.preds_per_test, shuffle=True, num_workers=4,
                                     drop_last=True, collate_fn=seg_test_collate)


# function set
funcs = FuncsWithConfig(config)
if 0.0<config.model.prob_mask_s<1.0:
    funcs.seg_train_dataloader = seg_train_dataloader
    funcs.seg_test_dataloader = seg_test_dataloader
    funcs.seg_train_iter = iter(seg_train_dataloader)
    funcs.seg_test_iter = iter(seg_test_dataloader)

# load the model
tags = funcs.get_tags()
folder_path = os.path.join('results', config.data.dataset.upper(), args.config_path.replace(".yaml", ""))
ckpt_path = os.path.join(folder_path, '-'.join(tags)+'.pt') 
states = torch.load(ckpt_path)  # [model_params, optimizer_params, epoch, step]
print(f"--------- load {ckpt_path} ---------------")
model = UNet_DDPM(config)

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

model.load_state_dict(states[0])
model.eval()

# some models for accuracies
model_lpips = eval_models.PerceptualLoss(model='net-lin',net='alex', device=config.device)
i3d = load_i3d_pretrained(device=config.device)    # make i3d model

# 評価するときは、1つの正解データに対してpreds_per_test回の予測を行う。
# そして、その平均値をとる。
# なので、正解データが5個であれば、モデルは5*preds_per_test回サンプリングを行い、その平均がスコアとなる。
result_base = {"mse": [],
                "ssim":[],
                "lpips":[],
                "embeddings":{"target":[], "pred":[]},}
################################################################
# TODO(?) calculate segmentation accuracy ######################
#result_seg_base = { "j mean":     
#}
###############################################################
result = {task: copy.deepcopy(result_base) for task in tags}
step = 0
for test_batch in tqdm(test_dataloader):
    #if step==1:
    #    break
    #else:
    #    step+=1
    with torch.no_grad():        
        test_batch = data_transform(config, test_batch)
        target, conds_test = funcs.separate_frames(test_batch.to(device))       # (B, F, C, H, W)
        target = target.reshape(target.shape[0], -1, target.shape[-2], target.shape[-1])    # (B, F*C, H, W)
            
        for task in tags:
            if task == "generation":
                prob_mask_p = 1.0
                prob_mask_f = 1.0
                prob_mask_s = 1.0
            elif task == "interpolation":
                prob_mask_p = 0.0
                prob_mask_f = 0.0
                prob_mask_s = 1.0
            elif task == "past_prediction":
                prob_mask_p = 1.0
                prob_mask_f = 0.0
                prob_mask_s = 1.0
            elif task == "future_prediction":
                prob_mask_p = 0.0
                prob_mask_f = 1.0
                prob_mask_s = 1.0
            elif task == "segmentation":
                prob_mask_p = 1.0
                prob_mask_f = 1.0
                prob_mask_s = 0.0
            
            masked_conds, masks = funcs.get_masked_conds(conds_test, prob_mask_p, prob_mask_f, prob_mask_s, mode='test')
            
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
                    
                # change input frames for segmentation
                if masks[2] is not None:
                    for index in range(len(target)):
                        if masks[2][index]==True:
                            # replace input frames to 'segmentation' from 'frame_generation'
                            target[index] = masked_conds[2][1][index].reshape(-1, target[index].shape[-2], target[index].shape[-1])   # seg_annotaion
            
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
                                            model_lpips=model_lpips, i3d=i3d, calc_fvd=False, only_embedding=True)
            
            # append each score
            for key in result_base.keys():
                if key=="embeddings":
                    result[task][key]["target"].append(accuracies[key][0])
                    result[task][key]["pred"].append(accuracies[key][1])
                else:
                    result[task][key].append(accuracies[key])

# save accuracies
with open(os.path.join(folder_path, f'[{args.config_path.replace(".yaml", "")}]_'+'-'.join(tags)+'_test_results.pkl'), "wb") as f:
    pickle.dump(result, f)

def get_avg_std_from_best_score_list(best_score_list):
    avg, std = best_score_list.mean().item(), best_score_list.std().item()
    return round(avg,3), round(std,3)

# calculate accuracies average, std, (conf95(= 95% confidence interval))   
print("======== calc avg, std ========")
calc_result = {}
for task in result.keys():
    calc_result[task] = {}
    print(f"----- ↓{task} -----")
    for key in result[task].keys():
        if key in ["mse", "ssim", "lpips"]:
            score_list = np.array(result[task][key]).reshape((-1, config.eval.preds_per_test))
            # get best score list
            if key=="mse":
                score_list = score_list.min(-1)
                # calc and save psnr_score
                psnr_list = (10 * np.log10(1 / np.array(result[task][key]))).reshape((-1, config.eval.preds_per_test)).max(-1)
                avg, std = get_avg_std_from_best_score_list(psnr_list)
                calc_result[task]["psnr"] = {"avg": avg, "std": std}
                print(f"psnr:\t{avg}±{std}")
            elif key=="ssim":
                score_list = score_list.max(-1)
            elif key=="lpips":
                score_list = score_list.min(-1)
            # get avg, std
            avg, std = get_avg_std_from_best_score_list(score_list)
            calc_result[task][key] = {"avg": avg, "std": std}
            print(f"{key}:\t{avg}±{std}")
            
        elif key=="fvd" and (result[task][key][0] is not None):
            # fvd per batch
            avg, std = get_avg_std_from_best_score_list(result[task][key])
            calc_result[task]["fvd_per_batch"] = {"avg":avg, "std":std}
            print(f"fvd_per_batch:\t{avg}±{std}")
            
        elif key=="embeddings":
            target_embeddings = np.concatenate(np.array(result[task][key]["target"]))
            pred_embeddings = np.concatenate(np.array(result[task][key]["pred"]))
            fvd = round(frechet_distance(pred_embeddings, target_embeddings),3)
            if config.eval.preds_per_test > 1:
                fvds_list = []
                # calc FVD for each trajectory and its average
                trajs = np.random.choice(np.arange(config.eval.preds_per_test), (config.eval.preds_per_test), replace=False)
                for traj in trajs:
                    fvds_list.append(frechet_distance(pred_embeddings[traj::config.eval.preds_per_test], target_embeddings))
                fvd_traj_avg, fvd_traj_std = round(float(np.mean(fvds_list)),3), round(float(np.std(fvds_list)),3)
                calc_result[task]["fvd_traj"] = {"avg":fvd_traj_avg, "std":fvd_traj_std}
                print(f"fvd_traj:\t{fvd_traj_avg}±{fvd_traj_std}")
                
            calc_result[task]["fvd"] = {"avg":fvd}
            print(f"fvd:\t{fvd}")

with open(os.path.join(folder_path, f'[{args.config_path.replace(".yaml", "")}]_'+'-'.join(tags)+'_test_calc_scores.txt'), "w") as f:
    for task in calc_result.keys():
        print(f"---{task}---", file=f)
        for key in calc_result[task].keys():
            print(f"{key}:\t{calc_result[task][key]}", file=f)


# plot generated video
with torch.no_grad():        
    test_batch = test_batch[::config.eval.preds_per_test]   # ignore the repeated ones
    
    for task in tags:
        target, conds_test = funcs.separate_frames(test_batch.to(device))       # (B, F, C, H, W)
        target = target.reshape(target.shape[0], -1, target.shape[-2], target.shape[-1])

        if task == "generation":
            prob_mask_p = 1.0
            prob_mask_f = 1.0
            prob_mask_s = 1.0
        elif task == "interpolation":
            prob_mask_p = 0.0
            prob_mask_f = 0.0
            prob_mask_s = 1.0
        elif task == "past_prediction":
            prob_mask_p = 1.0
            prob_mask_f = 0.0
            prob_mask_s = 1.0
        elif task == "future_prediction":
            prob_mask_p = 0.0
            prob_mask_f = 1.0
            prob_mask_s = 1.0
        elif task == "segmentation":
            prob_mask_p = 1.0
            prob_mask_f = 1.0
            prob_mask_s = 0.0
        
        masked_conds, masks = funcs.get_masked_conds(conds_test, prob_mask_p, prob_mask_f, prob_mask_s, mode='test')
        
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
                
            # change input frames for segmentation
            if masks[2] is not None:
                for index in range(len(target)):
                    if masks[2][index]==True:
                        # replace input frames to 'segmentation' from 'frame_generation'
                        target[index] = masked_conds[2][1][index].reshape(-1, target[index].shape[-2], target[index].shape[-1])   # seg_annotaion
                
        init_batch = funcs.get_init_sample(model, target.shape)
        # Get predicted x_0
        pred = funcs.reverse_process(model, init_batch, masked_conds_test, subsample_steps=config.eval.subsample_steps, final_only=True)  # pred : ['0' if final_only else 'len(steps)', B, C*F, H, W]
        pred = inverse_data_transform(config, pred[-1]).cpu()
        
        # 
        target = inverse_data_transform(config, target).cpu()
        #import matplotlib.pyplot as plt
        #plt.imshow(target[0].reshape(-1,3,target.shape[-2], target.shape[-1])[0].permute(1,2,0).to('cpu'))
        #plt.show()
        conds_for_plot = []
        for i, d in enumerate(masked_conds):
            if i==2:
                if d[0] is not None:
                    conds_for_plot.append(inverse_data_transform(config, d[0]).cpu())
                else:
                    conds_for_plot.append(None)
            else:
                if d is not None:
                    conds_for_plot.append(inverse_data_transform(config, d).cpu())
                else:
                    conds_for_plot.append(None)
        #masked_conds = [inverse_data_transform(config, d).cpu() if d is not None else None for d in masked_conds]

        funcs.plot_frames(conds_for_plot, [target, pred], video_folder=folder_path, task_name=task, config_filename=args.config_path.replace(".yaml", ""))