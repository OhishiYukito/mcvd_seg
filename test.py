from models.unet import UNet, UNet_SMLD, UNet_DDPM
from tools.functions_with_config import FuncsWithConfig
from datasets import get_dataset
from config import dict2namespace
from datasets import get_dataset, data_transform, inverse_data_transform

import torch
from torch.utils.data import DataLoader
import yaml
import os


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
for test_batch in test_dataloader:
    with torch.no_grad():        
        test_batch = data_transform(config, test_batch)
        
        target, conds_test = funcs.separate_frames(test_batch.to(device))
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
            accuracies = funcs.get_accuracy(pred, target, conds_test)
            
            # TODO save accuracies