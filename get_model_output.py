from config import dict2namespace
from datasets import get_dataset
import torch
from torch.utils.data import DataLoader
from models.unet import UNet_DDPM
from tools.functions_with_config import FuncsWithConfig

import argparse
import yaml
import os


torch.manual_seed(10)

# get args
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', help="path of config (.yaml)", default='bair_fppgs_deeper_9_9_7.yaml')
args = parser.parse_args()

# load config
with open('config/'+args.config_path) as f:
   dict_config = yaml.load(f, Loader=yaml.FullLoader)
config = dict2namespace(dict_config)

# def my_collate(batch):
#    #data = zip(*batch)
#    data = torch.stack(batch).repeat_interleave(config.eval.preds_per_test, dim=0)
#    #return data, torch.zeros(len(data))
#    return data

_, test_dataset = get_dataset(config)
test_dataloader = DataLoader(test_dataset, batch_size=getattr(config.train, 'batch_size', 64)//config.eval.preds_per_test, shuffle=True, num_workers=4, drop_last=True)
if 0.0<=config.model.prob_mask_s<1.0:
    seg_train_dataset, seg_test_dataset = get_dataset(config, segmentation=True)
    seg_train_dataloader = DataLoader(seg_train_dataset, batch_size=getattr(config.train, 'batch_size', 64), shuffle=True, num_workers=100, drop_last=True)
    def seg_test_collate(batch):
        origin_batch = torch.stack([data[0] for data in batch]).repeat_interleave(config.eval.preds_per_test, dim=0)
        ann_batch = torch.stack([data[1] for data in batch]).repeat_interleave(config.eval.preds_per_test, dim=0)
        return origin_batch, ann_batch
    seg_test_dataloader = DataLoader(seg_test_dataset, batch_size=getattr(config.train, 'batch_size', 64)//config.eval.preds_per_test, shuffle=True, num_workers=4,
                                     drop_last=True, collate_fn=seg_test_collate)


# function set
funcs = FuncsWithConfig(config)
if 0.0<=config.model.prob_mask_s<1.0:
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


for test_batch in test_dataloader:
    break

# plot generated video
with torch.no_grad():        
    #test_batch = test_batch[::config.eval.preds_per_test]   # ignore the repeated ones
    
    for task in tags:
        target_total, conds_test_first = funcs.separate_frames(test_batch.to(device), mode='test')       # (B, F, C, H, W)
        target_total = target_total.reshape(target_total.shape[0], -1, target_total.shape[-2], target_total.shape[-1])    # (B, F*C, H, W)

        # if task == "generation":
        #     prob_mask_p = 1.0
        #     prob_mask_f = 1.0
        #     prob_mask_s = 1.0
        # elif task == "interpolation":
        #     prob_mask_p = 0.0
        #     prob_mask_f = 0.0
        #     prob_mask_s = 1.0
        # elif task == "past_prediction":
        #     prob_mask_p = 1.0
        #     prob_mask_f = 0.0
        #     prob_mask_s = 1.0
        # elif task == "future_prediction":
        #     prob_mask_p = 0.0
        #     prob_mask_f = 1.0
        #     prob_mask_s = 1.0
        # elif task == "segmentation":
        #     prob_mask_p = 1.0
        #     prob_mask_f = 1.0
        #     prob_mask_s = 0.0
        if task == "segmentation":
            prob_mask_p = 1.0
            prob_mask_f = 1.0
            prob_mask_s = 1.0
        else:
            break        

        masked_conds_for_plot, _ = funcs.get_masked_conds(conds_test_first, prob_mask_p, prob_mask_f, prob_mask_s, mode='test')
        num_pred_on_step = config.data.num_frames
        channels = config.data.channels
        prev_pred = None
        all_pred = None
        # recursive to get (num_frames_total) frames.
        for num_step in range((config.data.num_frames_total+num_pred_on_step-1)//num_pred_on_step):
            target = target_total[:, channels*(num_step*num_pred_on_step) : channels*((num_step+1)*num_pred_on_step)]   # (B, C*F_on_step, H, W)
            if prev_pred is None:
                conds_test = conds_test_first
            else:
                # next line is executed only when future frames are not used.
                # so we can set 'conds_test = [cond_past, None(=cond_future)]'
                conds_test = [torch.cat((conds_test[0], prev_pred), dim=1)[:, -channels*config.data.num_frames_cond:], None]
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
                            target_total[index, channels*(num_step*num_pred_on_step) : channels*((num_step+1)*num_pred_on_step)] = target[index]
                            masked_conds_for_plot[2] = (masked_conds[2][0], masked_conds_for_plot[2][1])

            init_batch = funcs.get_init_sample(model, target.shape)
            # Get predicted x_0
            pred = funcs.reverse_process(model, init_batch, masked_conds_test, subsample_steps=config.eval.subsample_steps, final_only=True)  # pred : ['0' if final_only else 'len(steps)', B, C*F, H, W]
    
            prev_pred = pred[-1]
            #pred = inverse_data_transform(config, pred[-1]).cpu()
            pred  = pred[-1].cpu()
            
            #if task == "segmentation":
            #    # convert to 0 or 1
            #    pred = pred > 0.5
            #    pred = pred.to(torch.int32)
            if all_pred is None:
                all_pred = pred     # (B, C*F, H, W)
            else:
                all_pred = torch.cat((all_pred, pred), dim=1)

            # segmentationの場合は再帰させる必要がない
            if task=="segmentation":
                target_total = target
                break
            
        
        # cut off num_frames_total frames 
        if task!="segmentation":
            all_pred = all_pred[:, :config.data.num_frames_total*channels]
            target_total = target_total[:, :config.data.num_frames_total*channels]
    
        target_total = target_total.cpu()   #inverse_data_transform(config, target_total).cpu()
        #import matplotlib.pyplot as plt
        #plt.imshow(target[0].reshape(-1,3,target.shape[-2], target.shape[-1])[0].permute(1,2,0).to('cpu'))
        #plt.show()
        conds_for_plot = []
        for i, d in enumerate(masked_conds_for_plot):
            if i==2:
                if d[0] is not None:
                    conds_for_plot.append(d[0].cpu())
                    #conds_for_plot.append(inverse_data_transform(config, d[0]).cpu())
                else:
                    conds_for_plot.append(None)
            else:
                if d is not None:
                    conds_for_plot.append(d.cpu())
                    #conds_for_plot.append(inverse_data_transform(config, d).cpu())
                else:
                    conds_for_plot.append(None)
        #masked_conds = [inverse_data_transform(config, d).cpu() if d is not None else None for d in masked_conds]

        funcs.plot_frames(conds_for_plot, [target_total, all_pred], video_folder=folder_path, task_name=task, config_filename=args.config_path.replace(".yaml", ""))