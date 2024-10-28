# adopted from
# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# and
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# and
# https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py
#
# thanks!


import os
import math
import torch
import torch.nn as nn
import numpy as np
from einops import repeat
import importlib

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def tensor_erode(bin_img, ksize=5):
    # padding for keeping size
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)

    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)
    # B x C x H x W x k x k

    eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
    return eroded

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "squaredcos_cap_v2":  # used for karlo prior
        # return early
        return betas_for_alpha_bar(
            n_timestep,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out

def make_structure_adjustable_ddim_timesteps():
    
    structure_steps=35
    ori_structure_tau = 25
    ori_space=20
    oristeps=50
    ratio = structure_steps/ori_structure_tau
    newspace = ori_space / ratio
    newddimsteps=int(1000/newspace)

    # 
    texture_steps = 30 #oristeps-structure_steps
    ori_texture_steps = oristeps - ori_structure_tau
    ratio = ori_texture_steps/texture_steps
    new_texture_space = ori_space * ratio

    structure_ts= make_ddim_timesteps(ddim_discr_method="uniform", 
                                        num_ddim_timesteps=newddimsteps,
                                        num_ddpm_timesteps=1000,
                                        verbose=False
                                        )[:structure_steps]
    texture_ts= make_ddim_timesteps(ddim_discr_method="uniform", 
                                        num_ddim_timesteps=int(1000/new_texture_space),
                                        num_ddpm_timesteps=1000,
                                        verbose=False
                                        )[int(1000/new_texture_space)-texture_steps+1:]
    res = np.concatenate([structure_ts, texture_ts], axis=0)
    
    return res

def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad(), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

class Conv2d_kwargs(nn.Conv2d):
    def forward(self, x, **kwargs):
        return self._conv_forward(x, self.weight, self.bias)

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        # return nn.Conv2d(*args, **kwargs)
        return Conv2d_kwargs(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

index = 0
from PIL import Image
def vis_conv_feature(x, timestep=None):
    if timestep is not None:
        print(f'ts={timestep.item()}')
    global index
    x = x.mean(dim=1) # bhw
    x = x - x.min()
    x = x / x.max()
    img = Image.fromarray(np.array(x[0].detach().cpu() * 255, dtype=np.uint8))
    timestr=f"{timestep.item()}" if timestep is not None else "None"
    img.save(f'/apdcephfs_cq2/share_1290939/yingqinghe/results/stablediffusion/outputs/txt2img-samples/cfg9-sd-2-1-1024-debug/dilatedconv/vis_conv/002-normalconv/{index}-time{timestr}.jpg')
    index += 1


import torch.nn.functional as F
class DilatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                 enable_dilated_conv=True, erode=False,):
        super().__init__()
        self.dilation = dilation
        if isinstance(dilation, tuple) and (isinstance(dilation[0],float) or isinstance(dilation[1],float)):
            dilation = (math.ceil(dilation[0]), math.ceil(dilation[1]))
            self.fractional = True
        elif isinstance(dilation, float):
            dilation = math.ceil(dilation)
            self.fractional = True
        else:
            self.fractional = False
        
        # enable_dilated_conv=False

        if enable_dilated_conv:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1, dilation=1)
        self.conv = conv

        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.enable_dilated_conv = enable_dilated_conv
        self.dilation_real = max(dilation) if isinstance(dilation, tuple) else dilation
        self.erode = erode

    def forward(self, x, timestep=None,):
        b,c,h,w=x.shape
        size_out = (h//self.stride, w//self.stride)
        
        if self.fractional and self.enable_dilated_conv:
            if isinstance(self.dilation, tuple):
                dilation_max1, dilation_max2 = math.ceil(self.dilation[0]), math.ceil(self.dilation[1])
                ratio1, ratio2 = dilation_max1 / self.dilation[0], dilation_max2 / self.dilation[1]
                size=(math.ceil(h*ratio1), math.ceil(w*ratio2))
                # print(f'sf={(ratio1, ratio2)} x before = {x.shape}')
                # x = F.interpolate(x, scale_factor=(ratio1, ratio2), mode="bilinear") #
                # print(size)
                if ratio1 == 1 and ratio2 == 0:
                    self.fractional = False
                else:
                    x = F.interpolate(x, size=size, mode="bilinear") #
                # --------------------------------
                # feature dilation; dilation=0.5
                # b,c,h,w = x.shape
                # tmp = torch.zeros(b,c,h*2,w*2,device=x.device)
                # for i in range(h):
                #     for j in range(w):
                #         tmp[:,:, i*2+1, j*2+1] = x[:, :, i, j]
                # x = tmp.half()
                # print(f'x after = {x.shape}')
                # --------------------------------

            elif isinstance(self.dilation, int) or isinstance(self.dilation, float):
                dilation_max= math.ceil(self.dilation)
                ratio = dilation_max / self.dilation
                size=(math.ceil(h*ratio), math.ceil(w*ratio))
                # x = F.interpolate(x, scale_factor=ratio, mode="bilinear")
                if ratio == 1:
                    self.fractional = False
                else:
                    x = F.interpolate(x, size=size, mode="bilinear")
            # print(f'dilated conv, after interpolate x shape={x.shape},type= {x.dtype}')
        
        smooth_feat = False
        if smooth_feat:
            # conv = nn.Conv2d(c, c, 2, groups=c, dilation=2, padding=1, bias=False, padding_mode='replicate')
            # conv.weight = nn.Parameter(torch.ones([c, 1, 2, 2], device=x.device)/4.)
            # x = conv(x)
            size_ori = x.shape[2:]
            size_down = (int(x.shape[2] // 2), int(x.shape[3] // 2))
            x = F.interpolate(x, size=size_down, mode="nearest")
            x = F.interpolate(x, size=size_ori, mode="bilinear")

        x = self.conv(x)
        if self.erode:
            x = tensor_erode(x, ksize=9)
            print(f'x={x.shape}, erode, self.dilation_real={self.dilation_real}')

        if self.fractional and self.enable_dilated_conv:
            if isinstance(self.dilation, tuple):
                # x = F.interpolate(x, scale_factor=(1/ratio1, 1/ratio2), mode="bilinear")
                x = F.interpolate(x, size=size_out, mode="bilinear")
            elif isinstance(self.dilation, int) or isinstance(self.dilation, float):
                # x = F.interpolate(x, scale_factor=1/ratio, mode="bilinear")
                x = F.interpolate(x, size=size_out, mode="bilinear")
        # print(f'dilated conv, after interpolate 2 x shape={x.shape},type= {x.dtype}')

        # vis 
        # vis_conv_feature(x, timestep=timestep)
        return x
        


class DilatedTransposedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.conv = conv
    def forward(self, x):
        print(f'x before transpose {x.shape}')
        _,_,H,W = x.shape
        x = self.conv(x)
        print(f'x after transpose {x.shape}')
        b,c,h,w= x.shape
        l = (h-H)//2
        x =  x[:,:,l:l+H, l:l+H, ]
        print(f'x after select {x.shape}')
        return x

# def make_2d_dilate_conv(oriconv, *args, **kwargs):
def make_2d_dilate_conv(oriconv, *args, kernel_size=3, stride=1, padding=1, dilation=1, erode=False):
    # if dilation[0] < 1: # TODO, tuple type
    #     # stride = int(1/dilation) # 2
    #     # stride = 2 # pixel dilate
    #     dilation = 2
    #     padding = 0
    #     # import pdb;pdb.set_trace()
    #     conv = DilatedTransposedConv2d(oriconv.weight.shape[1], oriconv.weight.shape[0], 
    #         kernel_size=kernel_size, stride=dilation, padding=padding)
    #     conv.conv.weight = nn.Parameter(oriconv.weight.permute(1,0,2,3))
    #     conv.conv.bias = oriconv.bias
    # else:
    # conv = DilatedConv2d(*args, **kwargs)
    conv = DilatedConv2d(*args, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                         erode=erode,)
    conv.conv.weight = oriconv.weight
    conv.conv.bias = oriconv.bias
    
    return conv

def make_2d_dilate_conv_kernel1(oriconv, *args, kernel_size=3, stride=1, padding=1, dilation=1):
    # kernel=1 for small res
    conv = DilatedConv2d(*args, kernel_size=1, stride=stride, padding=0, dilation=1)
    conv.conv.weight = nn.Parameter(oriconv.weight.mean(dim=(2,3),keepdim=True))
    conv.conv.bias = oriconv.bias
    return conv

def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class HybridConditioner(nn.Module):

    def __init__(self, c_concat_config, c_crossattn_config):
        super().__init__()
        self.concat_conditioner = instantiate_from_config(c_concat_config)
        self.crossattn_conditioner = instantiate_from_config(c_crossattn_config)

    def forward(self, c_concat, c_crossattn):
        c_concat = self.concat_conditioner(c_concat)
        c_crossattn = self.crossattn_conditioner(c_crossattn)
        return {'c_concat': [c_concat], 'c_crossattn': [c_crossattn]}


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()
