#%%

from typing import Tuple, Sequence, Dict, Union, Optional
import numpy as np
import math
import torch
import torch.nn as nn
import collections
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

import os

from landing_env import LandingEnv




#%%import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os


# This function creates start/end indices for sampling from flattened data
def create_sample_indices(
        episode_ends: np.ndarray,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0
    ) -> np.ndarray:
    """
    Creates indices for sampling sequences from a dataset composed of multiple episodes.
    """
    indices = []
    for i in range(len(episode_ends)):
        start_idx = 0 if i == 0 else episode_ends[i-1]
        end_idx = episode_ends[i]
        ep_len = end_idx - start_idx

        min_start = -pad_before
        max_start = ep_len - sequence_length + pad_after

        # For each possible start of a sequence, create an index entry
        for idx in range(min_start, max_start + 1):
            buffer_start = max(idx, 0) + start_idx
            buffer_end = min(idx + sequence_length, ep_len) + start_idx

            # Handle padding by tracking offsets
            start_offset = buffer_start - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end

            sample_start_idx = start_offset
            sample_end_idx = sequence_length - end_offset

            indices.append([
                buffer_start, buffer_end,
                sample_start_idx, sample_end_idx
            ])
    return np.array(indices, dtype=int)

# This function samples a sequence and applies padding if necessary
def sample_sequence(
        train_data: dict,
        sequence_length: int,
        buffer_start_idx: int,
        buffer_end_idx: int,
        sample_start_idx: int,
        sample_end_idx: int
    ) -> dict:
    """
    Samples a sequence from the training data, padding where necessary.
    """
    result = {}
    for key, data_array in train_data.items():
        segment = data_array[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0 or sample_end_idx < sequence_length:
            # Create a padded array and fill it
            padded = np.zeros((sequence_length, *data_array.shape[1:]), dtype=data_array.dtype)
            # Pad with edge values
            if sample_start_idx > 0:
                padded[:sample_start_idx] = segment[0]
            if sample_end_idx < sequence_length:
                padded[sample_end_idx:] = segment[-1]
            padded[sample_start_idx:sample_end_idx] = segment
            result[key] = padded
        else:
            result[key] = segment
    return result

# Helper function for data statistics
def get_data_stats(data: np.ndarray) -> dict:
    """Computes min and max statistics for a given data array."""
    return {'min': data.min(axis=0), 'max': data.max(axis=0)}

# ====================================================================
# MODIFIED NORMALIZATION FUNCTIONS TO PREVENT ZeroDivisionError
# ====================================================================
def normalize_data(data: np.ndarray, stats: dict) -> np.ndarray:
    """
    Normalizes data to the range [-1, 1] robustly.
    Handles the case where a feature is constant across the dataset.
    """
    min_val = stats['min']
    max_val = stats['max']
    data_range = max_val - min_val
    
    # Create an output array, default to 0 for constant features
    norm_data = np.zeros_like(data, dtype=np.float32)

    # Identify non-constant features
    non_zero_range_mask = (data_range != 0)

    # Apply normalization only to non-constant features using broadcasting
    # data[..., non_zero_range_mask] selects all dimensions of data, but only the columns where the feature is not constant
    norm_data[..., non_zero_range_mask] = \
        2 * (data[..., non_zero_range_mask] - min_val[non_zero_range_mask]) / data_range[non_zero_range_mask] - 1
        
    return norm_data

def unnormalize_data(data: np.ndarray, stats: dict) -> np.ndarray:
    """
    Unnormalizes data from the range [-1, 1] back to its original scale robustly.
    """
    min_val = stats['min']
    max_val = stats['max']
    data_range = max_val - min_val
    
    # Map data from [-1, 1] to [0, 1]
    data_0_1 = (data + 1) / 2.0
    
    # For constant features, data_range is 0. The unnormalized value is just the min_val.
    # This formula works because for those features, data_0_1 * 0 + min_val = min_val.
    return data_0_1 * data_range + min_val
# ====================================================================

class LandingDataset(Dataset):
    """
    PyTorch Dataset for the aircraft landing trajectories.
    """
    def __init__(
        self,
        dataset_path: str,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
    ):
        # Load the dataset, allowing for object arrays due to variable lengths
        data = np.load(dataset_path, allow_pickle=True)
        X_episodes = data['X']
        U_episodes = data['U']
        C_episodes = data['centers']
        V_episodes = data['car_velocities']
        num_episodes = len(X_episodes)

        # --- Process and Flatten Data ---
        all_obs = []
        all_actions = []
        episode_ends = []
        current_len = 0

        print("Processing dataset episodes...")
        for i in range(num_episodes):
            X = X_episodes[i]
            centers = C_episodes[i].ravel() # Flatten obstacle centers to a 1D array (6*3=18)
            car_vel = V_episodes[i]

            # Create the context vector (obstacles + velocity)
            context = np.tile(np.append(centers, car_vel), (len(X), 1))

            # Augment observations with the context
            obs_augmented = np.hstack([X, context])
            all_obs.append(obs_augmented)

            # Actions are the original (un-augmented) state vectors
            all_actions.append(X)

            current_len += len(X)
            episode_ends.append(current_len)

        # Concatenate all episodes into flat arrays
        obs_flat = np.concatenate(all_obs, axis=0)
        action_flat = np.concatenate(all_actions, axis=0)
        train_data = {'obs': obs_flat, 'action': action_flat}

        episode_ends = np.array(episode_ends)

        # --- Indexing for Sequence Sampling ---
        self.indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        # --- Normalization ---
        self.stats = {k: get_data_stats(v) for k, v in train_data.items()}
        self.norm_data = {
            k: normalize_data(v, self.stats[k]) for k, v in train_data.items()
        }

        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.action_dim = train_data['action'].shape[-1]
        print("Dataset ready.")


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        buffer_start, buffer_end, sample_start, sample_end = self.indices[idx]

        # Sample a sequence from the normalized data
        sampled_seq = sample_sequence(
            train_data=self.norm_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start,
            buffer_end_idx=buffer_end,
            sample_start_idx=sample_start,
            sample_end_idx=sample_end
        )

        # Extract observation and action parts of the sequence
        obs_part = sampled_seq['obs'][:self.obs_horizon]
        action_part = sampled_seq['action'][self.obs_horizon : self.obs_horizon + self.action_horizon]

        return {
            'obs': torch.from_numpy(obs_part).float(),
            'action': torch.from_numpy(action_part).float()
        }


#%%
#@markdown ### **Dataset Demo**

# parameters
dataset_path = 'landing_mpc_dataset.npz'
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
    # Fetch a batch to inspect its structure and shape
    print("\n--- Inspecting a Batch from the DataLoader ---")
    batch = next(iter(dataloader))
    
    obs_dim = dataset.norm_data['obs'].shape[-1]
    action_dim = dataset.action_dim

    print(f"Shape of 'obs' tensor: {batch['obs'].shape}")
    print(f"Expected 'obs' shape: (batch_size, obs_horizon, {obs_dim})")
    
    print(f"\nShape of 'action' tensor: {batch['action'].shape}")
    print(f"Expected 'action' shape: (batch_size, action_horizon, {action_dim})")

    # Verify dimensions
    assert batch['obs'].shape == (256, obs_horizon, obs_dim)
    assert batch['action'].shape == (256, action_horizon, action_dim)

    print("\nVerification successful! The 'obs' tensor now includes obstacle and velocity data.")



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
action_dim = 6

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
device_name = 'cuda'
device = torch.device(device_name)
_ = noise_pred_net.to(device)
#%%
#@markdown ### **Training**
#@markdown
#@markdown Takes about an hour. If you don't want to wait, skip to the next cell
#@ma
TRAIN = False
if TRAIN:
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


    save_path = "saved_models"
    os.makedirs(save_path, exist_ok=True)

    # Save EMA-updated model (used for inference)
    ema.copy_to(noise_pred_net.parameters())  # ensure ema weights are copied
    torch.save(noise_pred_net.state_dict(), os.path.join(save_path, "ema_noise_pred_net.pth"))

    # Optional: Save optimizer and scheduler too (for checkpointing/resuming)
    torch.save({
        'model_state_dict': noise_pred_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'epoch': num_epochs,
    }, os.path.join(save_path, "checkpoint.pth"))



#%% Inference / Evaluation when TRAIN is False
import os
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt


# Load pretrained EMA weights
load_pretrained = not(TRAIN) 
if load_pretrained:
    ckpt_path = "saved_models/ema_noise_pred_net.pth"
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    ema_noise_pred_net = noise_pred_net
    ema_noise_pred_net.load_state_dict(state)
    # wrap in EMAModel to ensure EMA params are active
    ema = EMAModel(parameters=ema_noise_pred_net.parameters(), power=0.75)
    ema.copy_to(ema_noise_pred_net.parameters())
    noise_pred_net = ema_noise_pred_net
    noise_pred_net.eval()
    print("Pretrained weights loaded.")
else:
    print("Skipped pretrained weight loading.")



#%% Inference / Evaluation
import numpy as np
import torch
from collections import deque
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
# assume all imports and definitions: LandingEnv, ema_noise_pred_net,
# noise_scheduler, normalize_data, unnormalize_data, stats,
# pred_horizon, obs_horizon, action_horizon, action_dim,
# num_diffusion_iters, device

max_steps = 100
env       = LandingEnv()
env.dt = 0.5
obs, info = env.reset(), {}

obs_deque = deque([obs] * obs_horizon, maxlen=obs_horizon)
traj, rewards, actions_list = [obs.copy()], [], []
step_idx, done = 0, False

noise_scheduler.set_timesteps(num_diffusion_iters)
ema_noise_pred_net.eval()

if __name__ == "__main__":
    with torch.no_grad():  # disable grads for inference
        with tqdm(total=max_steps, desc="Eval LandingEnv") as pbar:
            while not done and step_idx < max_steps:
                # 1) Build conditioning
                obs_seq = np.stack(obs_deque)                            
                nobs    = normalize_data(obs_seq, stats['obs'])           
                tensor_obs = torch.from_numpy(nobs).float().to(device)    
                obs_cond   = tensor_obs.unsqueeze(0).flatten(start_dim=1)  

                # 2) Initialize noise
                naction = torch.randn((1, pred_horizon, action_dim),
                                      device=device, dtype=torch.float32)

                # 3) Reverse diffusion
                for k in noise_scheduler.timesteps:
                    noise_pred = ema_noise_pred_net(
                        sample     = naction,
                        timestep   = k,
                        global_cond= obs_cond
                    )
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample

                # 4) Unnormalize & slice
                na_np      = naction.squeeze(0).detach().cpu().numpy()   
                
                all_actions= unnormalize_data(na_np, stats['action'])  
                start      = obs_horizon - 1
                actions    = all_actions[start : start + action_horizon]
                actions_list.append(actions.copy())
                print('all_actions', all_actions)
                # print(actions)
                # 5) Execute actions
                for u in actions:
                    obs, reward, done, _ = env.step(u[0:3])
                    obs_deque.append(obs)
                    traj.append(obs.copy())
                    rewards.append(reward)
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
                    if done or step_idx >= max_steps or obs[2] < 1.0:
                        break
                if obs[2] < 1.0 and abs(obs[1]) < 1.0 and abs(obs[3]) < np.deg2rad(5):
                    print("Landing successful!")
                    break

    # 6) Plot results
    print("Total reward:", sum(rewards))
    traj = np.array(traj)
    t    = np.arange(len(traj)) * env.dt

    plt.figure(figsize=(5,4))
    plt.plot(traj[:,0], traj[:,1], '-o')
    plt.xlabel('x [m]'); plt.ylabel('y [m]'); plt.title('Ground Track'); plt.grid(True)

    plt.figure(figsize=(5,4))
    plt.plot(t, traj[:,2], '-o')
    plt.xlabel('time [s]'); plt.ylabel('h [m]'); plt.title('Altitude vs Time'); plt.grid(True)
    
    


    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot(traj[:,0], traj[:,1], traj[:,2])

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('h [m]')
    ax.set_title('3D Landing Trajectory')

    # make all axes use the same scale
    xr = traj[:,0].max() - traj[:,0].min()
    yr = traj[:,1].max() - traj[:,1].min()
    zr = traj[:,2].max() - traj[:,2].min()
    max_range = max(xr, yr, zr) / 2.0

    x_mid = (traj[:,0].max() + traj[:,0].min()) * 0.5
    y_mid = (traj[:,1].max() + traj[:,1].min()) * 0.5
    z_mid = (traj[:,2].max() + traj[:,2].min()) * 0.5

    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)

    plt.tight_layout()
    plt.show()

