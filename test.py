from models.unet import UNet, UNet_SMLD, UNet_DDPM
from tools.functions_with_config import FuncsWithConfig
from datasets import get_dataset
from config import dict2namespace
from datasets import get_dataset, data_transform, inverse_data_transform
from models.fvd.fvd import frechet_distance

import torch
from torch.utils.data import DataLoader
import yaml
import os
import numpy as np
import pickle

# other parameters
config_filename = 'sample.yaml'

# load config
with open('config/'+config_filename) as f:
   dict_config = yaml.load(f, Loader=yaml.FullLoader)
config = dict2namespace(dict_config)


def my_collate(batch):
    #data = zip(*batch)
    data = torch.stack(batch).repeat_interleave(config.eval.preds_per_test, dim=0)
    #return data, torch.zeros(len(data))
    return data

# make the Dataset (Dataloader)
# https://github.com/voletiv/mcvd-pytorch/blob/master/runners/ncsn_runner.py#L254
train_dataset, test_dataset = get_dataset(config)
##### TODO make Dataset and Dataloader for Segmentation #########################################
test_dataloader = DataLoader(test_dataset, batch_size=getattr(config.train, 'batch_size', 64)//config.eval.preds_per_test, shuffle=True, num_workers=4, drop_last=True, collate_fn=my_collate)

# function set
funcs = FuncsWithConfig(config)

# load the model
tags = funcs.get_tags()
folder_path = os.path.join('results', config.data.dataset.upper())
ckpt_path = os.path.join(folder_path, '-'.join(tags)+'.pt')
states = torch.load(ckpt_path)  # [model_params, optimizer_params, epoch, step]
print(f"--------- load {ckpt_path} ---------------")
model = UNet_DDPM(config)
model.load_state_dict(states[0])
model.eval()

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

# 評価するときは、1つの正解データに対してpreds_per_test回の予測を行う。
# そして、その平均値をとる。
# なので、正解データが5個であれば、モデルは5*preds_per_test回サンプリングを行い、その平均がスコアとなる。
result_base = {"mse": [],
                "ssim":[],
                "lpips_list":[],
                "target_embeddings":[],
                "pred_embeddings":[]}
result = {task: result_base.copy() for task in tags}
for test_batch in enumerate(test_dataloader):
    with torch.no_grad():        
        test_batch = data_transform(config, test_batch)
        
        target, conds_test = funcs.separate_frames(test_batch.to(device))       # (B, F, C, H, W)
        target = target.reshape(target.shape[0], -1, target.shape[-2], target.shape[-1])
        
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
            accuracies = funcs.get_accuracy(pred, target, conds_test, calc_fvd=False, only_embedding=True)
            
            # append each score
            for key in result_base.keys():
                result[task][key].append(accuracies[key])

# TODO save accuracies
with open(os.path.join(folder_path, '-'.join(tags)+'test_results.pkl')) as f:
    pickle.dump(result, f)

def get_avg_std_from_best_score_list(best_score_list):
    avg, std = best_score_list.mean().item(), best_score_list.std().item()
    return avg, std

# TODO calculate accuracies average, std, (conf95(= 95% confidence interval))   
print("======== calc avg, std ========")
calc_result = {}
for task in result.keys():
    calc_result[task] = {}
    print(f"----- ↓{task} -----")
    for key in enumerate(result[task].keys()):
        if key in ["mse", "ssim", "lpips"]:
            score_list = np.array(result[task][key]).reshape((-1, config.eval.preds_per_test))
            # get best score list
            if key=="mse":
                score_list = score_list.min(-1)
                # calc and save psnr_score
                psnr_list = (10 * np.log10(1 / np.array(result[task][key]))).reshape((-1, config.eval.preds_per_test).max(-1))
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
            target_embeddings = np.concatenate(result[task][key][0])
            pred_embeddings = np.concatenate(result[task][key][1])
            fvd = frechet_distance(pred_embeddings, target_embeddings)
            if config.eval.preds_per_test > 1:
                fvds_list = []
                # calc FVD for each trajectory and its average
                trajs = np.random.choice(np.arange(config.eval.preds_per_test), (config.eval.preds_per_test), replace=False)
                for traj in trajs:
                    fvds_list.append(frechet_distance(pred_embeddings[traj::config.eval.preds_per_test], target_embeddings))
                fvd_traj_avg, fvd_traj_std = float(np.mean(fvds_list)), float(np.std(fvds_list))
                calc_result[task]["fvd_traj"] = {"avg":fvd_traj_avg, "std":fvd_traj_std}
                print(f"fvd_traj:\t{fvd_traj_avg}±{fvd_traj_std}")
                
            calc_result[task]["fvd"] = {"avg":fvd}
            print(f"fvd:\t{avg}")

with open(os.path.join(folder_path, '-'.join(tags)+'test_calc_scores.txt'), "w") as f:
    for task in calc_result.keys():
        print(calc_result[task], file=f)