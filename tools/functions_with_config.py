# the class which has some functions with config for MCVD with Segmentation

import imageio
from tqdm import tqdm
import os
import torch
from torch.distributions.gamma import Gamma
import torch.nn.functional as F
import torchvision.transforms as Transforms
import numpy as np
from functools import partial
from torchvision.utils import make_grid, save_image
from skimage.metrics import structural_similarity as ssim
from cv2 import putText
from math import ceil

import models.eval_models as eval_models
from models.fvd.fvd import get_fvd_feats, load_i3d_pretrained, frechet_distance
from datasets import data_transform

class FuncsWithConfig:
    def __init__(self, config):
        # to separate frames into input and condition frames
        self.num_cond = config.data.num_frames_cond
        self.num_pred = config.data.num_frames
        self.num_future = config.data.num_frames_future
        assert self.num_cond>=0, f"num_frames_cond is {self.num_cond}! Set to [num_frames_cond >= 0] !"
        assert self.num_pred>=0, f"num_frames is {self.num_pred}! Set to [num_frames >= 0] !"
        assert self.num_future>=0, f"num_frames_future is {self.num_future}! Set to [num_frames_future >= 0] !"
        
        # to add noise to input frames
        self.num_forward_steps = config.model.num_forward_steps
        self.version = config.model.version
        
        # to make masked condition frames
        self.prob_mask_p = config.model.prob_mask_p
        self.prob_mask_f = config.model.prob_mask_f
        self.prob_mask_s = config.model.prob_mask_s
        assert self.prob_mask_p >= 0.0 and self.prob_mask_p <= 1.0, f"model.prob_mask_p is {self.prob_mask_p}! Set to 0.0<= prob_mask_p <=1.0 !"
        assert self.prob_mask_f >= 0.0 and self.prob_mask_f <= 1.0, f"model.prob_mask_f is {self.prob_mask_f}! Set to 0.0<= prob_mask_f <=1.0 !"
        assert self.prob_mask_s >= 0.0 and self.prob_mask_s <= 1.0, f"model.prob_mask_s is {self.prob_mask_s}! Set to 0.0<= prob_mask_s <=1.0 !" 

        self.config = config
        
    
    def separate_frames(self, batch):
        """separate frames into input and condition frames

        Args:
            batch (): [batch_size, num_total_frames, channel, size, size]
            
        return:
            x:  input frames ([batch_size, num_frames, channel, size, size])
            conds: condition frames ([conds_previous, conds_future]) 
                   (conds[i].shape = [batch_size, num_frames, channel, size, size])
        """
        cond_p = batch[:, :self.num_cond] if self.num_cond>0 else None
            
        x = batch[:, self.num_cond:self.num_cond+self.num_pred]
        
        cond_f = batch[:, self.num_cond+self.num_pred:] if self.num_future>0 else None
            
        conds = [cond_p, cond_f]
        
        return x, conds
        
        
    # predict the noise from input
    def get_noisy_frames(self, model, x, gamma=False):
        """forward steps which add noise determined by t to x

        Args:
            model : model which has attribute 'sigmas' or 'alphas' or more...
            x : input frames (x_0) [B, F, C, H, W]
            gamma : whether to use gamma distribution instead of normal distribution
        
        Returns:
            t : sampled timesteps
            z : the noises added to input x_0 [B, C*F, H, W]
            x_t (Tensor) : noisy input frames [B, C*F, H, W]
        """
        model = model.module if hasattr(model, 'module') else model
        
        # sampling timesteps
        t = torch.randint(0, self.num_forward_steps, (x.shape[0],), device=x.device)
        
        # add noise to input x
        # https://github.com/voletiv/mcvd-pytorch/blob/451da2eb635bad50da6a7c03b443a34c6eb08b3a/losses/dsm.py#L17
        if self.version == "SMLD":
            sigmas = model.sigmas
            used_sigmas = sigmas[t].reshape(x.shape[0], *([1] * len(x.shape[1:])))
            z = torch.randn_like(x).to(self.config.device)
            x_t = x + used_sigmas * z
        elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
            alphas = model.alphas
            used_alphas = alphas[t].reshape(x.shape[0], *([1] * len(x.shape[1:])))
            if gamma:
                # sampling noises from gamma distribution
                used_k = model.k_cum[t].reshape(x.shape[0], *([1] * len(x.shape[1:]))).repeat(1, *x.shape[1:])
                used_theta = model.theta_t[t].reshape(x.shape[0], *([1] * len(x.shape[1:]))).repeat(1, *x.shape[1:])
                z = Gamma(used_k, 1 / used_theta).sample().to(self.config.device)
                z = (z - used_k*used_theta) / (1 - used_alphas).sqrt()
            else:
                # sampling noises from normal distribution
                z = torch.randn_like(x).to(self.config.device)
            x_t = used_alphas.sqrt().to(x.device) * x + (1 - used_alphas).sqrt() * z
        
        z = z.reshape(z.shape[0], -1, z.shape[-2], z.shape[-1])
        x_t = x_t.reshape(x_t.shape[0], -1, x_t.shape[-2], x_t.shape[-1])
        
        return t, z, x_t
        
        
    # masking condition frames
    def get_masked_conds(self, conds, prob_mask_p=None, prob_mask_f=None, prob_mask_s=None, mode='train'):
        """conduct masking condition frames
            if use segmentation, its original frames are sampled in this function.\n 
            conds[0] and conds[1] will be reshaped in this function ([B, F, C, H, W]->[B, F*C, H, W])

        Args:
            conds: condition frames ([cond_previous, cond_future])
                   cond.shape = (batch, num_frames, C, H, W)
            prob_mask_p:
            prob_mask_f:
            prob_mask_s:
            mode: only be used with segmentation
        
        Returns:
            masked_conds: masked condition frames ([masked_previous, masked_future, (masked_seg, seg_ann)])
                          cond.shape = (batch, num_frames*C, H, W)
            masks: used masks (has values only if NOT deterministic (=> [0.0 < prob_mask < 1.0]))
        """
        # masking previous frames
        mask_p = None
        
        prob_mask_p = prob_mask_p if prob_mask_p is not None else self.prob_mask_p
        prob_mask_f = prob_mask_f if prob_mask_f is not None else self.prob_mask_f
        prob_mask_s = prob_mask_s if prob_mask_s is not None else self.prob_mask_s
        
        
        # Masking previous frames
        if self.num_cond>0 and conds[0] is not None:
            # (B, F, C, H, W) -> (B, F*C, H, W)
            conds[0] = conds[0].reshape(len(conds[0]), -1, conds[0].shape[-2], conds[0].shape[-1])   

            if prob_mask_p == 0.0:
                # NOT masking
                masked_cond = conds[0]
            elif 0.0 < prob_mask_p < 1.0:
                # random masking
                mask_p = torch.rand(conds[0].shape[0], device=conds[0].device) > prob_mask_p
                masked_cond = conds[0] * mask_p.reshape(-1, 1, 1, 1)
                mask_p = mask_p.to(torch.int32)
            elif prob_mask_p==1.0:
                # previous frames are not needed
                masked_cond = torch.zeros(conds[0].shape, device=conds[0].device)
        else:
            masked_cond = None
        
        
        # Masking future frames
        mask_f = None
        if self.num_future>0 and conds[1] is not None:
            # (B, F, C, H, W) -> (B, F*C, H, W)
            conds[1] = conds[1].reshape(len(conds[1]), -1, conds[0].shape[-2], conds[0].shape[-1])
            
            if prob_mask_f == 0.0:
                # NOT masking
                masked_future = conds[1]
            elif 0.0 <  prob_mask_f < 1.0:
                # random masking
                mask_f = torch.rand(conds[1].shape[0], device=conds[1].device) > prob_mask_f
                masked_future = conds[1] * mask_f.reshape(-1, 1, 1, 1)
                mask_f = mask_f.to(torch.int32)
            elif prob_mask_f==1.0:
                # future frames are not needed
                masked_future = torch.zeros(conds[1].shape, device=conds[1].device)
        else:
            masked_future = None
            
            
        # Masking segmentation frames
        if 0.0<=self.prob_mask_s<1.0:     
            # if conduct 'segmentatioin' in 'train', conds=[past, future, seg].
            # when 'frame_generation' in 'test', prob_mask_s was set to 1.0, but we have to set conds=[past, future, seg(zeros)] 
             
            # prepare the original data
            try:
                seg_origin, seg_ann = next(self.seg_train_iter) if mode=='train' else next(self.seg_test_iter)
            except StopIteration:
                if mode=='train':
                    self.seg_train_iter = iter(self.seg_train_dataloader)
                    seg_origin, seg_ann = next(self.seg_train_iter)
                elif mode=='test':
                    self.seg_test_iter = iter(self.seg_test_dataloader)
                    seg_origin, seg_ann = next(self.seg_test_iter)
            seg_origin, seg_ann = data_transform(self.config, seg_origin), data_transform(self.config, seg_ann)
            # (B,F,C,H,W) -> (B, F*C, H, W)
            seg_origin = seg_origin.reshape(len(seg_origin), -1, seg_origin.shape[-2], seg_origin.shape[-1]).to(self.config.device)
            seg_ann = seg_ann.reshape(len(seg_ann), -1, seg_ann.shape[-2], seg_ann.shape[-1]).to(self.config.device)
            
            if 0.0<=prob_mask_s<1.0:
                mask_s = torch.rand(seg_origin.shape[0], device=seg_origin.device) > prob_mask_s
                if prob_mask_s==0.0:
                    # NOT masking
                    masked_seg = seg_origin  # frames before be segmented
                else:
                    # random masking    
                    masked_seg = seg_origin * mask_s.reshape(-1, 1, 1, 1)
                    mask_s = mask_s.to(torch.int32)
            elif prob_mask_s==1.0:
                # masking
                mask_s, seg_ann = None, None
                masked_seg = torch.zeros(seg_origin.shape, device=seg_origin.device)
                
        elif self.prob_mask_s==1.0:
            # NOT use segmentation
            mask_s, masked_seg, seg_ann = None, None, None
            
            
        # When 'segmentation', 'frame generation task' is deactivate
        if mask_s is not None:
            inversed_mask_s = torch.tensor(list(map(lambda x: not x, mask_s))).to(device=self.config.device)
            if masked_cond is not None:
                masked_cond = masked_cond * inversed_mask_s.reshape(-1,1,1,1)
                mask_p = inversed_mask_s.to(torch.int32)
            if masked_future is not None:
                masked_future = masked_future * inversed_mask_s.reshape(-1,1,1,1)
                mask_f = inversed_mask_s.to(torch.int32)
        
        return [masked_cond, masked_future, (masked_seg, seg_ann)], [mask_p, mask_f, mask_s]
    
  
    
    # get tags representing tasks that the model can perform
    def get_tags(self):
        """ get tags
        
        Returns:
            tags (list): tag lists for wandb.init()
        """
        tags = []
        if self.num_cond==0:
            if self.num_future==0 or self.prob_mask_f>0.0:
                tags.append("generation")
            if self.num_future>0 and (0.0<=self.prob_mask_f<1.0):
                tags.append("past_prediction")
        else:
            if self.num_future==0:
                if 0<=self.prob_mask_p<1:
                    tags.append("generation")
                if 0<self.prob_mask_p<=1:
                    tags.append("future_prediction")
            else:
                if 0<=self.prob_mask_p<1 and 0<=self.prob_mask_f<1:
                    tags.append("interpolation")
                if 0<=self.prob_mask_p<1 and 0<self.prob_mask_f<=1:
                    tags.append("future_prediction")
                if 0<self.prob_mask_p<=1 and 0<=self.prob_mask_f<1:
                    tags.append("past_prediction")
                if 0<self.prob_mask_p<=1 and 0<self.prob_mask_f<=1:
                    tags.append("generation")
        if 0.0<=self.prob_mask_s<1.0:
            tags.append("segmentation")
        
        return tags
                
                
    def get_init_sample(self, model, target_shape):
        """ get init sample that is first input for reverse process

        Args:
            model (_type_): to get param 'k_cum', 'theta_t' in case of "gamma"==True
            target_shape (_type_): output shape

        Returns:
            z : init sample from normal distribution
        """
        if self.config.model.version == "SMLD":
            z = torch.rand(target_shape, device=self.config.device)
            #z = data_transform(config, z)
        elif self.config.model.version == "DDPM" or self.config.model.version == "DDIM" or self.config.model.version == "FPNDM":
            if getattr(self.config.model, 'gamma', False):
                used_k, used_theta = model.k_cum[0], model.theta_t[0]
                z = Gamma(torch.full(target_shape, used_k), torch.full(target_shape, 1 / used_theta)).sample().to(self.config.device)
                z = z - used_k*used_theta
            else:
                z = torch.randn(target_shape, device=self.config.device)
                
        return z
    
    
    # https://github.com/voletiv/mcvd-pytorch/blob/master/models/__init__.py#L207
    def reverse_process(self, model, x_t, masked_cond, gamma=False, subsample_steps=None, opposite_schedule=False, clip_before=True, final_only=False):
        """ conduct reverse process

        Args:
            model (): diffusion
            x_t (): input batch from get_init_sample
            masked_cond : masked condition
            gamma (bool): whether to use gamma distribution
            subsample_steps (int): number of reverse process steps 
            opposite_schedule (bool): whether to use schedule opposite to forward process
            clip_before (bool) : whether to clip the calculated x_0 to [-1, 1] before calculate x_t-1
            final_only (bool) : whether to get only [x_0] or [x_T, x_T-1, ..., x_1, x_0] 

        Returns:
            x0: estimated target frames
        """
        model.eval()
        net = model.module if hasattr(model, 'module') else model
        alphas, alphas_prev, betas = net.alphas, net.alphas_prev, net.betas
        steps = np.arange(len(betas))
        if gamma:
            ks_cum, thetas = net.k_cum, net.theta_t
        
        if subsample_steps is not None:
            if subsample_steps < len(alphas):
                skip = len(alphas) // subsample_steps
                steps = range(0, len(alphas), skip)
                steps = torch.tensor(steps, device=alphas.device)
                # new alpha, beta, alpha_prev
                alphas = alphas.index_select(0, steps)
                alphas_prev = torch.cat([alphas[1:], torch.tensor([1.0]).to(alphas)])
                betas = 1.0 - torch.div(alphas, alphas_prev) # for some reason we lose a bit of precision here
                if gamma:
                    ks_cum = ks_cum.index_select(0, steps)
                    thetas = thetas.index_select(0, steps)
            
        images = []
        model = partial(model, cond=masked_cond)
        #x_transf = False
        
        for i, step in enumerate(tqdm(steps)):
        #for i, step in enumerate(steps):
            if opposite_schedule:
                i = len(betas)-1-i    # to calculate x_t, use alpha_T-t, beta_T-t
            
            timesteps = (step * torch.ones(x_t.shape[0], device=x_t.device))#.long()
            # Get model prediction
            pred_z = model(x_t, timesteps)
            
            # Estimate x_0 with (x_t, pred_z)
            c_beta, c_alpha, c_alpha_prev = betas[i], alphas[i], alphas_prev[i]
            x_0 = (1 / c_alpha.sqrt()) * (x_t - (1 - c_alpha).sqrt() * pred_z)
            if clip_before:
                # x_0 is treated as a frame, so the data range should be aligned with the frame data.
                x_0 = x_0.clip_(-1, 1)
                
            # Estimate x_t-1 with (x_0, x_t)
            x_t = (c_alpha_prev.sqrt() * c_beta / (1 - c_alpha)) * x_0 + ((1 - c_beta).sqrt() * (1 - c_alpha_prev) / (1 - c_alpha)) * x_t
            if not final_only:
                images.append(x_t.to('cpu'))
            
            # Add noise to x_t-1
            # - If last step (x_t-1 == x_0), don't add noise
            last_step = i + 1 == len(steps)
            if last_step:
                continue
            # - Else, add noise
            if gamma:
                z = Gamma(torch.full(x_t.shape[1:], ks_cum[i]),
                        torch.full(x_t.shape[1:], 1 / thetas[i])).sample((x_t.shape[0],)).to(x_t.device)
                noise = (z - ks_cum[i]*thetas[i])/((1 - alphas[i]).sqrt())
            else:
                noise = torch.randn_like(x_t)
            #if just_beta:
            #    x_mod += c_beta.sqrt() * noise
            #else:
            #    x_mod += ((1 - c_alpha_prev) / (1 - c_alpha) * c_beta).sqrt() * noise
            x_t += ((1 - c_alpha_prev) / (1 - c_alpha) * c_beta).sqrt() * noise
        
        # Denoise
        # if NCSN ( https://github.com/voletiv/mcvd-pytorch/issues/13 )
        #if denoise:
        #    last_noise = ((len(steps) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device))#.long()
        #    x_mod = x_mod - (1 - alphas[-1]).sqrt() * model(x_t, last_noise)
        #    if not final_only:
        #        images.append(x_mod.to('cpu'))

        if final_only:
            return x_t.unsqueeze(0)
        else:
            return torch.stack(images)

    
    
                
    def get_accuracy(self, pred_batch, target_batch, conds, model_lpips=None, i3d=None, calc_fvd=True, only_embedding=True):
        """ calculate MSE, SSIM, LPIPS, FVD of SingleBatch.
            Please conduct 'inverse_data_transform(batch)' before pass them to this function. 

        Args:
            pred_batch (): [B, C*F, H, W]
            target_batch (): [B, C*F, H, W]
            conds (): condition frames without mask (conds=[cond_p, cond_f], cond_p.shape = [B, F, C, H, W])
            calc_fvd (bool): whether to calculate fvd
            only_embedding: 
        
        Returns:
            accuracies (dict): key=["mse", "ssim", "lpips", "fvd", "embeddings"],
                                "fvd"(the score per batch) is float or None, 
                                "embeddings"=[target_embed, pred_embed] (each embed is a list that has embedding vector as elements, or None).
                                other params are list that has single video's score as an element. 
        """
        vid_mse, vid_ssim, vid_lpips = [], [], []
        
        T2 = Transforms.Compose([Transforms.Resize((128, 128)),
                                 Transforms.ToTensor(),
                                 Transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                      std=(0.5, 0.5, 0.5))])
        if model_lpips is None:
            model_lpips = eval_models.PerceptualLoss(model='net-lin',net='alex', device=self.config.device)
        
        
        if target_batch.shape[1] < pred_batch.shape[1]: # We cannot calculate MSE, PSNR, SSIM
            #print("-------- Warning: Cannot calculate metrics because predicting beyond the training data range --------")
            for ii in range(len(pred_batch)):
                vid_mse.append(0)
                vid_ssim.append(0)
                vid_lpips.append(0)
        else:
            # Calculate MSE, PSNR, SSIM
            for ii in range(len(pred_batch)):
                # per video
                mse, avg_ssim, avg_distance = 0, 0, 0
                for jj in range(self.num_pred):
                    # per frame
                    # MSE (and PSNR)
                    pred_ij = pred_batch[ii, (self.config.data.channels*jj):(self.config.data.channels*jj + self.config.data.channels), :, :]
                    target_ij = target_batch[ii, (self.config.data.channels*jj):(self.config.data.channels*jj + self.config.data.channels), :, :]
                    mse += F.mse_loss(target_ij, pred_ij)

                    pred_ij_pil = Transforms.ToPILImage()(pred_ij).convert("RGB")
                    target_ij_pil = Transforms.ToPILImage()(target_ij).convert("RGB")

                    # SSIM
                    pred_ij_np_grey = np.asarray(pred_ij_pil.convert('L'))  # L: gray scale (0~255)
                    target_ij_np_grey = np.asarray(target_ij_pil.convert('L'))
                    if self.config.data.dataset.upper() == "STOCHASTICMOVINGMNIST" or self.config.data.dataset.upper() == "MOVINGMNIST":
                        # ssim is the only metric extremely sensitive to gray being compared to b/w 
                        pred_ij_np_grey = np.asarray(Transforms.ToPILImage()(torch.round(pred_ij)).convert("RGB").convert('L'))
                        target_ij_np_grey = np.asarray(Transforms.ToPILImage()(torch.round(target_ij)).convert("RGB").convert('L'))
                    avg_ssim += ssim(pred_ij_np_grey, target_ij_np_grey, data_range=255, gaussian_weights=True, use_sample_covariance=False)

                    # LPIPS
                    pred_ij_LPIPS = T2(pred_ij_pil).unsqueeze(0).to(self.config.device)
                    target_ij_LPIPS = T2(target_ij_pil).unsqueeze(0).to(self.config.device)
                    avg_distance += model_lpips.forward(target_ij_LPIPS, pred_ij_LPIPS)

                vid_mse.append((mse / self.num_pred).to('cpu'))
                vid_ssim.append(avg_ssim / self.num_pred)
                vid_lpips.append(avg_distance.data.item() / self.num_pred)
        
        
        # FVD
        if calc_fvd or only_embedding:
            # concat past + current + future frames            
            if conds[0] is not None:
                conds[0] = conds[0].reshape(len(conds[0]), -1, conds[0].shape[-2], conds[0].shape[-1])
                target_videos = torch.cat([conds[0], target_batch], dim=1) 
                pred_videos = torch.cat([conds[0], pred_batch], dim=1)
            else:
                target_videos = target_batch
                pred_videos = pred_batch
            if conds[1] is not None:
                conds[1] = conds[1].reshape(len(conds[1]), -1, conds[1].shape[-2], conds[1].shape[-1])
                target_videos = torch.cat([target_videos, conds[1]], dim=1)
                pred_videos = torch.cat([pred_videos, conds[1]], dim=1)
                
            #target_videos = target_videos[::self.config.eval.preds_per_test]    # ignore the repeated ones
                
            # convert video_frames shape (B, F, C, H, W) or (B, F*C, H, W) -> (B, C, F, H, W)
            def to_i3d(x):
                x = x.reshape(x.shape[0], -1, self.config.data.channels, x.shape[-2], x.shape[-1])
                if self.config.data.channels == 1:
                    x = x.repeat(1, 1, 3, 1, 1) # hack for greyscale images
                x = x.permute(0, 2, 1, 3, 4)  # BTCHW -> BCTHW
                return x
            
            # make i3d model
            if i3d is None:
                i3d = load_i3d_pretrained(device=self.config.device)
            
            # get i3d embedding (use 'get_fvd_feats()' )
            target_videos, pred_videos = to_i3d(target_videos), to_i3d(pred_videos)
            target_feats = get_fvd_feats(videos=target_videos, i3d=i3d, device=self.config.device)
            target_feats = target_feats[::self.config.eval.preds_per_test]      # ignore the repeated ones
            pred_feats = get_fvd_feats(videos=pred_videos, i3d=i3d, device=self.config.device)
            
            
            # get frechet distance between target_video and predicted_video (use 'frechet_distance()')
            target_feats, pred_feats = np.array(target_feats), np.array(pred_feats) 
            vid_fvd = frechet_distance(pred_feats, target_feats) if calc_fvd else None
        
        else:
            vid_fvd, target_feats, pred_feats = None, None, None
            
        return {"mse":vid_mse, "ssim":vid_ssim, "lpips":vid_lpips, "fvd":vid_fvd, "embeddings":[target_feats, pred_feats] }
    
    
    # https://github.com/voletiv/mcvd-pytorch/blob/master/runners/ncsn_runner.py#L1996
    def plot_frames(self, conds, current_frames, video_folder, task_name, config_filename):
        """save frames as gif and png

        Args:
            conds (list): condition frames [past, future]   (B, F*C, H, W)
            current_frames (list): [target, predicted]      (B, F*C, H, W)
            video_folder (str): path to save gif
            task_name (str): be used as a part of filename
            config_filename (str): be used as a part of filename (REMOVE extension '.yaml')
        """
   
        gif_frames_cond = []
        gif_frames_pred= []
        gif_frames_futr = []

        # we show conditional frames, and real&pred side-by-side
        # past frames
        if conds[0] is not None and torch.count_nonzero(conds[0])!=0:
            # there are past frames, and they are not be masked
            cond = conds[0]
            for t in range(conds[0].shape[1]//self.config.data.channels):
                cond_t = cond[:, t*self.config.data.channels:(t+1)*self.config.data.channels]     # BCHW
                frame = torch.cat([cond_t, 0.5*torch.ones(*cond_t.shape[:-1], 2), cond_t], dim=-1)
                frame = frame.permute(0, 2, 3, 1).numpy()
                frame = np.stack([putText(f.copy(), f"{t+1:2d}p", (4, 15), 0, 0.5, (1,1,1), 1) for f in frame])
                nrow = ceil(np.sqrt(2*cond.shape[0])/2)
                gif_frame = make_grid(torch.from_numpy(frame).permute(0, 3, 1, 2), nrow=nrow, padding=6, pad_value=0.5).permute(1, 2, 0).numpy()  # HWC
                gif_frames_cond.append((gif_frame*255).astype('uint8'))
                if t == 0:
                    gif_frames_cond.append((gif_frame*255).astype('uint8'))
                del frame, gif_frame
                
        # current frames
        real = current_frames[0]
        pred = current_frames[1]
        if real.shape[1] < pred.shape[1]: # Pad with zeros to prevent bugs
            real = torch.cat([real, torch.zeros(real.shape[0], pred.shape[1]-real.shape[1], real.shape[2], real.shape[3])], dim=1)
         
        for t in range(pred.shape[1]//self.config.data.channels):
            real_t = real[:, t*self.config.data.channels:(t+1)*self.config.data.channels]     # BCHW
            pred_t = pred[:, t*self.config.data.channels:(t+1)*self.config.data.channels]     # BCHW
            frame = torch.cat([real_t, 0.5*torch.ones(*pred_t.shape[:-1], 2), pred_t], dim=-1)
            frame = frame.permute(0, 2, 3, 1).numpy()
            frame = np.stack([putText(f.copy(), f"{t+1:02d}", (4, 15), 0, 0.5, (1,1,1), 1) for f in frame])
            nrow = ceil(np.sqrt(2*pred.shape[0])/2)
            gif_frame = make_grid(torch.from_numpy(frame).permute(0, 3, 1, 2), nrow=nrow, padding=6, pad_value=0.5).permute(1, 2, 0).numpy()  # HWC
            gif_frames_pred.append((gif_frame*255).astype('uint8'))
            if t == pred.shape[1]//self.config.data.channels - 1 and (conds[1] is None):
                gif_frames_pred.append((gif_frame*255).astype('uint8'))
            del frame, gif_frame
        
        # future frames
        if conds[1] is not None and torch.count_nonzero(conds[1])!=0:
            # there are future frames, and they are not be masked
            futr = conds[1]
            for t in range(futr.shape[1]//self.config.data.channels):
                futr_t = futr[:, t*self.config.data.channels:(t+1)*self.config.data.channels]     # BCHW
                frame = torch.cat([futr_t, 0.5*torch.ones(*futr_t.shape[:-1], 2), futr_t], dim=-1)
                frame = frame.permute(0, 2, 3, 1).numpy()
                frame = np.stack([putText(f.copy(), f"{t+1:2d}f", (4, 15), 0, 0.5, (1,1,1), 1) for f in frame])
                nrow = ceil(np.sqrt(2*futr.shape[0])/2)
                gif_frame = make_grid(torch.from_numpy(frame).permute(0, 3, 1, 2), nrow=nrow, padding=6, pad_value=0.5).permute(1, 2, 0).numpy()  # HWC
                gif_frames_futr.append((gif_frame*255).astype('uint8'))
                if t == futr.shape[1]//self.config.data.channels - 1:
                    gif_frames_futr.append((gif_frame*255).astype('uint8'))
                del frame, gif_frame
        
        #if conds[2] is not None and torch.count_nonzero(conds[2])!=0:
            # there are segmentation frames, and they are not be masked
            
                
        # Save gif
        if task_name=="future_prediction":          # Future Prediction
            imageio.mimwrite(os.path.join(video_folder, f"[{config_filename}]_videos_future-pred.gif"),
                                [*gif_frames_cond, *gif_frames_pred], duration=1000 * 1/4, loop=0)
        elif task_name=="interpolation":            # Interpolation
            imageio.mimwrite(os.path.join(video_folder, f"[{config_filename}]_videos_interp.gif"),
                                [*gif_frames_cond, *gif_frames_pred, *gif_frames_futr], duration=1000 * 1/4, loop=0)
        elif task_name=="generation":               # Generation
            imageio.mimwrite(os.path.join(video_folder, f"[{config_filename}]_videos_gen.gif"),
                                gif_frames_pred, duration=1000 * 1/4, loop=0)
        elif task_name=="past_prediction":          # Past Prediction
            imageio.mimwrite(os.path.join(video_folder, f"[{config_filename}]_videos_past-pred.gif"),
                                [*gif_frames_pred, *gif_frames_futr], duration=1000 * 1/4, loop=0)
        elif task_name=='segmentation':
            ###TODO make segmentation gif #########################################
            pass

        del gif_frames_cond, gif_frames_pred, gif_frames_futr
        
        # Save stretch frames
        def stretch_image(X, ch, imsize):
            # (B, F*C, H, W) -> (B, C, H, F*W)
            return X.reshape(len(X), -1, ch, imsize, imsize).permute(0, 2, 1, 4, 3).reshape(len(X), ch, -1, imsize).permute(0, 1, 3, 2)
        
        def save_past_pred(pred, real):
            # save frames as dict
            torch.save({"futr": futr, "pred": pred, "real": real},
                        os.path.join(video_folder, f"[{config_filename}]_videos_past-pred.pt"))
            pred_im = stretch_image(pred, self.config.data.channels, pred.shape[-1])
            real_im = stretch_image(real, self.config.data.channels, real.shape[-1])
            futr_im = stretch_image(futr, self.config.data.channels, futr.shape[-1])
            padding_hor = 0.5*torch.ones(*real_im.shape[:-1], 2)
            real_data = torch.cat([real_im, padding_hor, futr_im], dim=-1)
            pred_data = torch.cat([pred_im, padding_hor, 0.5*torch.ones_like(futr_im)], dim=-1)
            padding_ver = 0.5*torch.ones(*real_im.shape[:-2], 2, real_data.shape[-1])
            data = torch.cat([real_data, padding_ver, pred_data], dim=-2)
            # Save stretched image
            # Set 'nrow' as follows to make the image shape closer to a square
            nrow = ceil(np.sqrt((self.num_pred+self.num_future)*pred.shape[0])/(self.num_pred+self.num_future))
            image_grid = make_grid(data, nrow=nrow, padding=6, pad_value=0.5)
            save_image(image_grid, os.path.join(video_folder, f"[{config_filename}]_videos_stretch_past-pred.png"))
        
        def save_future_pred(pred, real):
            # save frames as dict
            torch.save({"cond": cond, "pred": pred, "real": real},
                        os.path.join(video_folder, f"[{config_filename}]_videos_future-pred.pt"))
            cond_im = stretch_image(cond, self.config.data.channels, cond.shape[-1])
            pred_im = stretch_image(pred, self.config.data.channels, pred.shape[-1])
            real_im = stretch_image(real, self.config.data.channels, real.shape[-1])
            padding_hor = 0.5*torch.ones(*real_im.shape[:-1], 2)
            real_data = torch.cat([cond_im, padding_hor, real_im], dim=-1)
            pred_data = torch.cat([0.5*torch.ones_like(cond_im), padding_hor, pred_im], dim=-1)
            padding_ver = 0.5*torch.ones(*real_im.shape[:-2], 2, real_data.shape[-1])
            data = torch.cat([real_data, padding_ver, pred_data], dim=-2)
            # Save stretched image
            # Set 'nrow' as follows to make the image shape closer to a square
            nrow = ceil(np.sqrt((self.num_cond+self.num_pred)*pred.shape[0])/(self.num_cond+self.num_pred))
            image_grid = make_grid(data, nrow=nrow, padding=6, pad_value=0.5)
            save_image(image_grid, os.path.join(video_folder, f"[{config_filename}]_videos_stretch_future-pred.png"))

        def save_interp(pred, real):
            # save frames as dict
            torch.save({"cond": cond, "pred": pred, "real": real, "futr": futr},
                            os.path.join(video_folder, f"[{config_filename}]_videos_interp.pt"))
            cond_im = stretch_image(cond, self.config.data.channels, cond.shape[-1])
            pred_im = stretch_image(pred, self.config.data.channels, pred.shape[-1])
            real_im = stretch_image(real, self.config.data.channels, real.shape[-1])
            futr_im = stretch_image(futr, self.config.data.channels, futr.shape[-1])
            padding_hor = 0.5*torch.ones(*real_im.shape[:-1], 2)
            real_data = torch.cat([cond_im, padding_hor, real_im, padding_hor, futr_im], dim=-1)
            pred_data = torch.cat([0.5*torch.ones_like(cond_im), padding_hor, pred_im, padding_hor, 0.5*torch.ones_like(futr_im)], dim=-1)
            padding_ver = 0.5*torch.ones(*real_im.shape[:-2], 2, real_data.shape[-1])
            data = torch.cat([real_data, padding_ver, pred_data], dim=-2)
            # Save stretched image
            # Set 'nrow' as follows to make the image shape closer to a square
            nrow = ceil(np.sqrt((self.num_cond+self.num_pred+self.num_future)*pred.shape[0])/(self.num_cond+self.num_pred+self.num_future))
            image_grid = make_grid(data, nrow=nrow, padding=6, pad_value=0.5)
            save_image(image_grid, os.path.join(video_folder, f"[{config_filename}]_videos_stretch_interp.png"))

        def save_gen(pred):
            if pred is None:
                return
            # save frames as dict
            torch.save({"gen": pred}, os.path.join(video_folder, f"[{config_filename}]_videos_gen.pt"))
            data = stretch_image(pred, self.config.data.channels, pred.shape[-1])
            # Save stretched image
            # Set 'nrow' as follows to make the image shape closer to a square
            nrow = ceil(np.sqrt((self.num_pred)*pred.shape[0])/(self.num_pred))
            image_grid = make_grid(data, nrow=nrow, padding=6, pad_value=0.5)
            save_image(image_grid, os.path.join(video_folder, f"[{config_filename}]_videos_stretch_gen.png"))

        if task_name=="past_prediction":
            save_past_pred(pred, real)

        elif task_name=="future_prediction":
            save_future_pred(pred, real)
            
        elif task_name=="interpolation":
            save_interp(pred, real)

        elif task_name=="generation":
            save_gen(pred)