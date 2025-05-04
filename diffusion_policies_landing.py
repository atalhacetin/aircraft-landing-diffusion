#%%

from typing import Tuple, Sequence, Dict, Union, Optional
import numpy as np
import math
import torch
import torch.nn as nn
import collections
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import torch.multiprocessing as mp

# BEFORE you create any DataLoaders:
mp.set_sharing_strategy('file_system')

# env import
import gym
from gym import spaces
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from skvideo.io import vwrite
import os




#%%
import numpy as np
import torch
from torch.utils.data import Dataset

def create_sample_indices(
        episode_ends: np.ndarray,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0
    ) -> np.ndarray:
    indices = []
    for i in range(len(episode_ends)):
        start_idx = 0 if i == 0 else episode_ends[i-1]
        end_idx   = episode_ends[i]
        ep_len    = end_idx - start_idx

        min_start = -pad_before
        max_start = ep_len - sequence_length + pad_after

        for idx in range(min_start, max_start + 1):
            buf_start = max(idx, 0) + start_idx
            buf_end   = min(idx + sequence_length, ep_len) + start_idx
            start_off = buf_start - (idx + start_idx)
            end_off   = (idx + sequence_length + start_idx) - buf_end
            sample_start = start_off
            sample_end   = sequence_length - end_off
            indices.append([
                buf_start, buf_end,
                sample_start, sample_end
            ])
    return np.array(indices, dtype=int)

def sample_sequence(
        train_data: dict,
        sequence_length: int,
        buffer_start_idx: int,
        buffer_end_idx: int,
        sample_start_idx: int,
        sample_end_idx: int
    ) -> dict:
    result = {}
    for key, arr in train_data.items():
        seg = arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0 or sample_end_idx < sequence_length:
            padded = np.zeros((sequence_length, *arr.shape[1:]), dtype=arr.dtype)
            if sample_start_idx > 0:
                padded[:sample_start_idx] = seg[0]
            if sample_end_idx < sequence_length:
                padded[sample_end_idx:] = seg[-1]
            padded[sample_start_idx:sample_end_idx] = seg
            result[key] = padded
        else:
            result[key] = seg
    return result

def get_data_stats(data: np.ndarray) -> dict:
    flat = data.reshape(-1, data.shape[-1])
    return {'min': flat.min(axis=0), 'max': flat.max(axis=0)}

def normalize_data(data: np.ndarray, stats: dict) -> np.ndarray:
    n = (data - stats['min']) / (stats['max'] - stats['min'])
    return n * 2 - 1

class LandingDataset(Dataset):
    """
    Dataset for .npz trajectories:
      X: (E, T+1, 6) full state
      U: (E, T,   3) controls (3D position)

    Returns fixed-length segments:
      'obs':    (obs_horizon,   6)
      'action': (pred_horizon,  3)
    """
    def __init__(
        self,
        dataset_path: str,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
    ):
        data = np.load(dataset_path)
        X = data['X']    # (E, T+1, 6)
        U = data['U']    # (E, T,   3)
        E, T1, sdim = X.shape
        T = T1 - 1       # original control length

        # pad U to length T+1 by repeating last control
        U_pad = np.concatenate([U, U[:,-1:,:]], axis=1)  # (E, T+1, 3)

        # flatten episodes
        obs_flat = X.reshape(E*(T+1), sdim)       # (E*(T+1), 6)
        act_flat = U_pad.reshape(E*(T+1), 3)      # (E*(T+1), 3)

        train_data = {'obs': obs_flat, 'action': act_flat}
        episode_ends = np.arange(1, E+1) * (T+1)

        # sample indices
        self.indices = create_sample_indices(
            episode_ends    = episode_ends,
            sequence_length = pred_horizon,
            pad_before      = obs_horizon - 1,
            pad_after       = action_horizon - 1,
        )

        # stats & normalize
        self.stats = {}
        self.norm_data = {}
        for key, arr in train_data.items():
            st = get_data_stats(arr)
            self.stats[key] = st
            self.norm_data[key] = normalize_data(arr, st)

        self.pred_h = pred_horizon
        self.obs_h  = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        bs, be, ss, se = self.indices[idx]
        seq = sample_sequence(
            train_data      = self.norm_data,
            sequence_length = self.pred_h,
            buffer_start_idx = bs,
            buffer_end_idx   = be,
            sample_start_idx = ss,
            sample_end_idx   = se,
        )
        seq['obs'] = seq['obs'][:self.obs_h]
        return {
            'obs':    torch.from_numpy(seq['obs']).float(),
            'action': torch.from_numpy(seq['action']).float(),
        }

# Example usage:
# ds = LandingNPZDataset('landing_param_dataset.npz', pred_horizon=50, obs_horizon=10, action_horizon=50)
# loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)



#%%
#@markdown ### **Dataset Demo**

# parameters
dataset_path = 'landing_param_dataset.npz'
pred_horizon = 16
obs_horizon = 2
action_horizon = 8
#|o|o|                             observations: 2
#| |a|a|a|a|a|a|a|a|               actions executed: 8
#|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

# create dataset from file
dataset = LandingDataset(
    dataset_path=dataset_path,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)
# save training data statistics (min, max) for each dim
stats = dataset.stats

# create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    num_workers=0,
    shuffle=True,
    # accelerate cpu-gpu transfer
    #pin_memory=True,
    # don't kill worker process afte each epoch
    #persistent_workers=True
)


if __name__ == "__main__":

    # visualize data in batch
    batch = next(iter(dataloader))
    print("batch['obs'].shape:", batch['obs'].shape)
    print("batch['action'].shape", batch['action'].shape)


#%%

#@markdown ### **Network**
#@markdown
#@markdown Defines a 1D UNet architecture `ConditionalUnet1D`
#@markdown as the noies prediction network
#@markdown
#@markdown Components
#@markdown - `SinusoidalPosEmb` Positional encoding for the diffusion iteration k
#@markdown - `Downsample1d` Strided convolution to reduce temporal resolution
#@markdown - `Upsample1d` Transposed convolution to increase temporal resolution
#@markdown - `Conv1dBlock` Conv1d --> GroupNorm --> Mish
#@markdown - `ConditionalResidualBlock1D` Takes two inputs `x` and `cond`. \
#@markdown `x` is passed through 2 `Conv1dBlock` stacked together with residual connection.
#@markdown `cond` is applied to `x` with [FiLM](https://arxiv.org/abs/1709.07871) conditioning.

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x


#%%
#@markdown ### **Network Demo**

# observation and action dimensions corrsponding to
# the output of PushTEnv
obs_dim = 6
action_dim = 3

# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)

# example inputs
noised_action = torch.randn((1, pred_horizon, action_dim))
obs = torch.zeros((1, obs_horizon, obs_dim))
diffusion_iter = torch.zeros((1,))

# the noise prediction network
# takes noisy action, diffusion iteration and observation as input
# predicts the noise added to action
noise = noise_pred_net(
    sample=noised_action,
    timestep=diffusion_iter,
    global_cond=obs.flatten(start_dim=1))

# illustration of removing noise
# the actual noise removal is performed by NoiseScheduler
# and is dependent on the diffusion noise schedule
denoised_action = noised_action - noise

# for this demo, we use DDPMScheduler with 100 diffusion iterations
num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

# device transfer
device_name = 'mps'
device = torch.device(device_name)
_ = noise_pred_net.to(device)
#%%
#@markdown ### **Training**
#@markdown
#@markdown Takes about an hour. If you don't want to wait, skip to the next cell
#@markdown to load pre-trained weights

num_epochs = 100

# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights
ema = EMAModel(
    parameters=noise_pred_net.parameters(),
    power=0.75)

# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
optimizer = torch.optim.AdamW(
    params=noise_pred_net.parameters(),
    lr=1e-4, weight_decay=1e-6)

# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

with tqdm(range(num_epochs), desc='Epoch') as tglobal:
    # epoch loop
    for epoch_idx in tglobal:
        epoch_loss = list()
        # batch loop
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                # data normalized in dataset
                # device transfer
                nobs = nbatch['obs'].to(device)
                naction = nbatch['action'].to(device)
                B = nobs.shape[0]

                # observation as FiLM conditioning
                # (B, obs_horizon, obs_dim)
                obs_cond = nobs[:,:obs_horizon,:]
                # (B, obs_horizon * obs_dim)
                obs_cond = obs_cond.flatten(start_dim=1)

                # sample noise to add to actions
                noise = torch.randn(naction.shape, device=device)

                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=device
                ).long()

                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = noise_scheduler.add_noise(
                    naction, noise, timesteps)

                # predict the noise residual
                noise_pred = noise_pred_net(
                    noisy_actions, timesteps, global_cond=obs_cond)

                # L2 loss
                loss = nn.functional.mse_loss(noise_pred, noise)

                # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                ema.step(noise_pred_net.parameters())

                # logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)
        tglobal.set_postfix(loss=np.mean(epoch_loss))

# Weights of the EMA model
# is used for inference
ema_noise_pred_net = noise_pred_net
ema.copy_to(ema_noise_pred_net.parameters())



#%%
#@markdown ### **Loading Pretrained Checkpoint**
#@markdown Set `load_pretrained = True` to load pretrained weights.

  #@markdown ### **Inference**

# limit enviornment interaction to 200 steps before termination
# max_steps = 200
# env = PushTEnv()
# # use a seed >200 to avoid initial states seen in the training dataset
# env.seed(100000)

# # get first observation
# obs, info = env.reset()

# # keep a queue of last 2 steps of observations
# obs_deque = collections.deque(
#     [obs] * obs_horizon, maxlen=obs_horizon)
# # save visualization and rewards
# imgs = [env.render(mode='rgb_array')]
# rewards = list()
# done = False
# step_idx = 0

# if __name__=="__main__":

#     with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
#         while not done:
#             B = 1
#             # stack the last obs_horizon (2) number of observations
#             obs_seq = np.stack(obs_deque)
#             # normalize observation
#             nobs = normalize_data(obs_seq, stats=stats['obs'])
#             # device transfer
#             nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

#             # infer action
#             with torch.no_grad():
#                 # reshape observation to (B,obs_horizon*obs_dim)
#                 obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

#                 # initialize action from Guassian noise
#                 noisy_action = torch.randn(
#                     (B, pred_horizon, action_dim), device=device)
#                 naction = noisy_action

#                 # init scheduler
#                 noise_scheduler.set_timesteps(num_diffusion_iters)

#                 for k in noise_scheduler.timesteps:
#                     # predict noise
#                     noise_pred = ema_noise_pred_net(
#                         sample=naction,
#                         timestep=k,
#                         global_cond=obs_cond
#                     )

#                     # inverse diffusion step (remove noise)
#                     naction = noise_scheduler.step(
#                         model_output=noise_pred,
#                         timestep=k,
#                         sample=naction
#                     ).prev_sample

#             # unnormalize action
#             naction = naction.detach().to('cpu').numpy()
#             # (B, pred_horizon, action_dim)
#             naction = naction[0]
#             action_pred = unnormalize_data(naction, stats=stats['action'])

#             # only take action_horizon number of actions
#             start = obs_horizon - 1
#             end = start + action_horizon
#             action = action_pred[start:end,:]
#             # (action_horizon, action_dim)

#             # execute action_horizon number of steps
#             # without replanning
#             for i in range(len(action)):
#                 # stepping env
#                 obs, reward, done, _, info = env.step(action[i])
#                 # save observations
#                 obs_deque.append(obs)
#                 # and reward/vis
#                 rewards.append(reward)
#                 imgs.append(env.render(mode='rgb_array'))

#                 # update progress bar
#                 step_idx += 1
#                 pbar.update(1)
#                 pbar.set_postfix(reward=reward)
#                 if step_idx > max_steps:
#                     done = True
#                 if done:
#                     break

#     # print out the maximum target coverage
#     print('Score: ', max(rewards))

#     #vwrite('vis.mp4', imgs)
#     video_array = np.stack(imgs, axis=0).astype(np.uint8)

#     # write with H.264 libx264, 30 FPS, yuv420p pixel format for widest compatibility
#     vwrite(
#         "vis.mp4",
#         video_array,
#         outputdict={
#             '-r': '30',                # fps
#             '-vcodec': 'libx264',      # encoder
#             '-pix_fmt': 'yuv420p'      # pixel format
#         }
#     )