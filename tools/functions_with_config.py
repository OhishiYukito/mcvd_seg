# the class which has some functions with config for MCVD with Segmentation


import torch
from torch.distributions.gamma import Gamma
import torch.nn.functional as F
import torchvision.transforms as Transforms
import numpy as np
from skimage.metrics import structural_similarity as ssim
from functools import partial

from models.fvd.fvd import get_fvd_feats, load_i3d_pretrained, frechet_distance
import models.eval_models as eval_models

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
        
        cond_f = batch[:, self.num_cond+self.num_pred:] if self.num_future else None
            
        conds = [cond_p, cond_f]
        
        return x, conds
        
        
    # predict the noise from input
    def get_noisy_frames(self, model, x, gamma=False):
        """forward steps which add noise determined by t to x

        Args:
            model : model which has attribute 'sigmas' or 'alphas' or more...
            x : input frames (x_0)
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
    def get_masked_conds(self, conds, prob_mask_p=None, prob_mask_f=None):
        """conduct masking condition frames
            # conds[0] and conds[1] will be reshaped in this function ([B, F, C, H, W]->[B, F*C, H, W])

        Args:
            conds: condition frames ([cond_previous, cond_future])
                   cond.shape = (batch, num_frames, C, H, W)
        
        Returns:
            masked_conds: masked condition frames ([masked_previous, masked_future])
                          cond.shape = (batch, num_frames*C, H, W)
            masks: used masks (has values only if NOT deterministic (=> [0.0 < prob_mask < 1.0]))
        """
        # masking previous frames
        mask_p = None
        
        prob_mask_p = prob_mask_p if prob_mask_p is not None else self.prob_mask_p
        prob_mask_f = prob_mask_f if prob_mask_f is not None else self.prob_mask_f

        if self.num_cond>0 and conds[0] is not None:
            # (batch_size, num_frames, C, H, W) -> (batch_size, num_frames*C, H, W)
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
        
        # masking future frames
        mask_f = None
        if self.num_future>0 and conds[1] is not None:
            # (batch_size, num_frames, C, H, W) -> (batch_size, num_frames*C, H, W)
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
            
        # masking segmentation frames

        return [masked_cond, masked_future], [mask_p, mask_f]
    
    
    #def plot_image(self):
    
    
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
        
        for i, step in enumerate(steps):
            if opposite_schedule:
                step = len(betas)-1-step    # to calculate x_t, use alpha_T-t, beta_T-t
            
            timesteps = (step * torch.ones(x_t.shape[0], device=x_t.device))#.long()
            # Get model prediction
            pred_z = model(x_t, timesteps)
            
            # Estimate x_0 with (x_t, pred_z)
            c_beta, c_alpha, c_alpha_prev = betas[step], alphas[step], alphas_prev[step]
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

    
    
                
    def get_accuracy(self, pred_batch, target_batch, conds):
        """ calculate MSE, SSIM, LPIPS, FVD of SingleBatch

        Args:
            pred_batch (): [B, C*F, H, W]
            target_batch (): [B, C*F, H, W]
            conds (): condition frames without mask (conds=[cond_p, cond_f], cond_p.shape = [B, F, C, H, W])
        
        Returns:
            accuracies : 
        """
        vid_mse, vid_ssim, vid_lpips = [], [], []
        
        T2 = Transforms.Compose([Transforms.Resize((128, 128)),
                                 Transforms.ToTensor(),
                                 Transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                      std=(0.5, 0.5, 0.5))])
        
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

                vid_mse.append(mse / self.num_pred)
                vid_ssim.append(avg_ssim / self.num_pred)
                vid_lpips.append(avg_distance.data.item() / self.num_pred)
        
        
        # FVD
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
            
        target_videos = target_videos[::self.config.eval.preds_per_test]    # ignore the repeated ones
            
        # convert video_frames shape (B, F, C, H, W) or (B, F*C, H, W) -> (B, C, F, H, W)
        def to_i3d(x):
            x = x.reshape(x.shape[0], -1, self.config.data.channels, x.shape[-2], x.shape[-1])
            if self.config.data.channels == 1:
                x = x.repeat(1, 1, 3, 1, 1) # hack for greyscale images
            x = x.permute(0, 2, 1, 3, 4)  # BTCHW -> BCTHW
            return x
        
        # make i3d model
        i3d = load_i3d_pretrained(device=self.config.device)
        
        # get i3d embedding (use 'get_fvd_feats()' )
        target_feats = get_fvd_feats(videos=to_i3d(target_videos), i3d=i3d, device=self.config.device)
        pred_feats = get_fvd_feats(videos=to_i3d(pred_videos), i3d=i3d, device=self.config.device)
        
        
        # get frechet distance between target_video and predicted_video (use 'frechet_distance()')
        target_feats, pred_feats = np.array(target_feats), np.array(pred_feats) 
        
        vid_fvd = frechet_distance(pred_feats, target_feats)
        
        return vid_mse, vid_ssim, vid_lpips, vid_fvd