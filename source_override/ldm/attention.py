from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any

from .util import checkpoint

import numpy as np
from PIL import Image
import cv2

def make_grid_np(imgs):
    """ List[np.array] with array size = [H,W,3] """
    n = len(imgs)
    n_cols=int(n**0.5)
    n_rows=int(np.ceil(n/n_cols))
    npad = n_cols * n_rows - n
    blankmaps = [np.zeros_like(imgs[0])] * npad
    rows=[]
    for i in range(n_rows):
        row = cv2.hconcat(imgs[i*n_cols:(i+1)*n_cols])
        if npad != 0 and i == n_rows-1:
            row = cv2.hconcat([row] + blankmaps)
        rows.append(row)
    imgsheet = cv2.vconcat(rows)
    return imgsheet

def vis_attention(attn, num_head, res, is_self, count, res_dir, save_single=False):
    os.makedirs(res_dir, exist_ok=True)
    token_number = attn.shape[-1] # bh, l, l | bh, 4096, 77
    h, w = res
    # assert(attn.shape[0] == num_head)
    if attn.shape[0] >= num_head:
        bs = attn.shape[0] // num_head
    else:
        bs= attn.shape[0]
    print(f'bs={bs}, realbs={attn.shape[0]}, numheads={num_head}')
    
    for bi in range(bs):
        if attn.shape[0] >= num_head:
            attn_ = attn[bi:(bi+1)*num_head, ...]
        elif attn.shape[0] < num_head:
            attn_ = attn[bi:bi+1, ...]
        typestr = "self" if is_self else "cross"
        maps=[]
        for ti in range(token_number):
            attn_vec = attn_[:, :, ti] # bh, l
            # assert(torch.sum(attn_vec, dim=1).all()==1)
            attn_vec = attn_vec.mean(dim=0) # 对head维度求mean
            image = rearrange(attn_vec, '(h w) -> h w', h=h, w=w) # 0-1
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.cpu().numpy()
            # ----------------------------------------------------------------
            image = image - image.min()
            image = image / image.max() # norm
            image = 255 * image
            # ----------------------------------------------------------------
            maps.append(image)
            if save_single:
                image = 255 * image / image.max()
                image = image.astype(np.uint8)
                image = np.array(Image.fromarray(image).resize((256, 256)))
                # image = vis.text_under_image(image, f'token {ti:02d}')
                subdir=os.path.join(res_dir, "single_token")
                os.makedirs(subdir, exist_ok=True)
                # fpath = os.path.join(subdir, f'bi{bi:02d}_attn_layer{layer_id:02d}_{typestr}_token{ti:02d}.jpg')
                fpath = os.path.join(subdir, f'bi{bi:02d}_attn_count{count:02d}_{typestr}_token{ti:02d}.jpg')
                Image.fromarray(image).save(fpath)
        maps = np.stack(maps, axis=0) # n,h,w,3
        # ----------------------------------------------------------------
        # maps = maps - maps.min()
        # maps = maps / maps.max() # norm
        # maps = 255 * maps
        # ----------------------------------------------------------------
        # maps = 255 * maps / maps.max() # max norm
        # ----------------------------------------------------------------
        maps = maps.astype(np.uint8)
        maps = make_grid_np(maps)
        subdir=os.path.join(res_dir, "all_token")
        os.makedirs(subdir, exist_ok=True)
        fpath = os.path.join(subdir, f'bi{bi:02d}_count{count:02d}_{typestr}.jpg')
        Image.fromarray(maps).save(fpath)

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.,
                 use_dialated_selfattn=False, 
                 attention_sf=False,
                 gen_size=None,
                 **kwargs):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.use_dialated_selfattn = use_dialated_selfattn
        self.count = 0
        self.attention_sf = attention_sf
        self.gen_size = gen_size

    def forward(self, x, context=None, mask=None):
        is_self = (context is None)
        if self.use_dialated_selfattn:
            return self.forward_dilated_self(x, context)
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        if self.attention_sf:
            # print('change attn sf')
            if is_self:
                test_res = self.gen_size[0] * self.gen_size[1] #1024 ** 2
                train_res = 512 ** 2
                test_train_ratio = test_res / train_res
                
                test_tn = q.shape[1]
                train_tn = test_tn / test_train_ratio
                sf = math.log(test_tn, train_tn) ** 0.5
                scale = sf * self.scale
            else:
                scale = self.scale
        else:
            scale = self.scale
        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * scale
        
        del q, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

    def forward_dilated_self(self, x, context=None):
        type = 'self' if context is None else 'cross'
        print(f'[INFO] use dilated attn, {type} attn')
        
        b,l,c=x.shape
        x = x.contiguous()
        h=int(l**0.5)
        H=h
        w=h
        x_ = rearrange(x, 'b (h w) c -> b c h w',h=H, w=w).contiguous()
        dilate_slices = [
            make_dilate_index(x_, offset=(0,0), flatten=True),
            make_dilate_index(x_, offset=(0,1), flatten=True),
            make_dilate_index(x_, offset=(1,0), flatten=True),
            make_dilate_index(x_, offset=(1,1), flatten=True),
        ]

        x = torch.cat([x[:, slice, :] for slice in dilate_slices], dim=0) # (n b) l c
        # x = x.reshape(4, b, l//4, c).contiguous().permute(1,0,2,3).contiguous().reshape(b*4, l//4, c).contiguous()# blc

        # normal forward
        # ----------------------------------------------------------------------------------
        h = self.heads
        q = self.to_q(x)
        if context is not None:
            context = context.repeat(4,1,1)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h).contiguous(), (q, k, v))
        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        del q, k
        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h).contiguous()
        out = self.to_out(out) # blc
        # vis_attention(out, self.heads, (H//2,w//2), is_self=True, count=self.count,
        #               res_dir="/apdcephfs_cq2/share_1290939/yingqinghe/results/stablediffusion/outputs/txt2img-samples/cfg9-sd-2-1-1024-debug/visattn/small", 
        #               save_single=False)
        # ----------------------------------------------------------------------------------

        # merge out
        out = rearrange(out, 'b (h w) c -> b c h w', h=H//2, w=w//2).contiguous()
        # out = out.reshape(b, c, 2, 2, H//2, w//2).contiguous()
        # out = out.permute(0, 1, 4, 2, 5, 3).contiguous() # b,c,h,2,w,2
        # out = out.reshape(b, c, H, w).contiguous()
        # out = rearrange(out, 'b c h w -> b (h w) c',h=H,w=w).contiguous()
        
        out = out.view(2, 2, b, c, H//2, w//2)
        out = out.permute(2, 3, 4, 0, 5, 1).contiguous() # b,c,h,2,w,2
        out = out.view(b, c, H, w)
        out = rearrange(out, 'b c h w -> b (h w) c',h=H,w=w).contiguous()

        # vis_attention(out, self.heads, (H,w), is_self=True, count=self.count,
        #               res_dir="/apdcephfs_cq2/share_1290939/yingqinghe/results/stablediffusion/outputs/txt2img-samples/cfg9-sd-2-1-1024-debug/visattn/merge", 
        #               save_single=False)
        
        self.count += 1
        return out
    
class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, 
                 use_dialated_selfattn=False,
                 attention_sf=False,
                 gen_size=None,
                 **kwargs):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None
        self.use_dialated_selfattn = use_dialated_selfattn
        self.attention_sf = attention_sf
        self.gen_size = gen_size

    def forward(self, x, context=None, mask=None):
        is_self = (context is None)
        if self.use_dialated_selfattn: # and (context is not None)
            print(f'[INFO] only cross dilated')
            return self.forward_dilated_self(x, context)
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )
        if self.attention_sf and is_self:
            print('[INFO] xformer change attn sf')
            test_res = self.gen_size[0] * self.gen_size[1] #1024 ** 2
            train_res = 512 ** 2
            test_train_ratio = test_res / train_res
            
            test_tn = q.shape[1]
            train_tn = test_tn / test_train_ratio
            sf = math.log(test_tn, train_tn) ** 0.5
            scale = sf * self.scale
        else:
            scale = None
        
        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, 
                                                      attn_bias=None, 
                                                      op=self.attention_op,
                                                      scale=scale,)
        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)

    def forward_dilated_self(self, x, context=None, mask=None):
        is_self = context is None

        if self.attention_sf:
            raise NotImplementedError
        
        type = 'self' if context is None else 'cross'
        print(f'use dilated attn, {type} attn')
        
        # dilate x
        B,l,c = x.shape
        x = x.contiguous()
        h=int(l**0.5)
        H=h
        w=h
        x_ = rearrange(x, 'b (h w) c -> b c h w',h=H, w=w).contiguous()
        dilate_slices = [
            make_dilate_index(x_, offset=(0,0), flatten=True),
            make_dilate_index(x_, offset=(0,1), flatten=True),
            make_dilate_index(x_, offset=(1,0), flatten=True),
            make_dilate_index(x_, offset=(1,1), flatten=True),
        ]
        x = torch.cat([x[:, slice, :] for slice in dilate_slices], dim=0) # (n b) l c

        # normal forward
        q = self.to_q(x)
        if context is not None:
            context = context.repeat(4,1,1)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)
        if exists(mask):
            raise NotImplementedError
        # -----------------------------------
        # adjust the attention scaling factor
        # -----------------------------------
        # if is_self:
        #     test_res = 256
        #     train_res = 512
        #     test_train_ratio = test_res / train_res
            
        #     test_tn = q.shape[1]
        #     train_tn = test_tn / (test_train_ratio ** 2)
        #     sf = math.log(test_tn, train_tn) # ** 0.5
        #     print(f'q={q.shape}, test_tn={test_tn}, train_tn={train_tn}, sf={sf}')
        #     out = out * sf
        # -----------------------------------
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        out = self.to_out(out)

        # merge out
        out = rearrange(out, 'b (h w) c -> b c h w', h=H//2, w=w//2).contiguous()
        out = out.view(2, 2, B, c, H//2, w//2)
        out = out.permute(2, 3, 4, 0, 5, 1).contiguous() # b,c,h,2,w,2
        out = out.view(B, c, H, w)
        out = rearrange(out, 'b c h w -> b (h w) c',h=H,w=w).contiguous()
        return out

class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention,
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False, use_dialated_selfattn=False, attention_sf=False, 
                 gen_size=(1024, 1024), 
                 **kwargs):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None, 
                              use_dialated_selfattn=use_dialated_selfattn,
                              attention_sf=attention_sf,
                              gen_size=gen_size,
                              **kwargs)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout,
                              use_dialated_selfattn=use_dialated_selfattn,
                              attention_sf=attention_sf,
                              gen_size=gen_size,
                              **kwargs)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


def make_dilate_index(x, dilate=1, offset=(0, 0), flatten=False):
    assert(x.dim() == 4)
    b,c,h,w=x.shape
    index = torch.zeros(h, w)
    dia_win = dilate + 1
    nwins = math.ceil(max(h/dia_win, w/dia_win))

    for ni in range(nwins):
      for nj in range(nwins):
        i = ni * dia_win + offset[0]
        j = nj * dia_win + offset[1]
        index[i, j] = 1
    
    if flatten:
        index = rearrange(index, 'h w -> (h w)').contiguous()
    return index.bool()


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True,
                 use_dialated_selfattn=False, 
                 **kwargs,
                 ):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint,
                                    use_dialated_selfattn=use_dialated_selfattn, **kwargs,)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None, **kwargs):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
           
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


    def forward_dilated_self(self, x, context=None):
        """both self & cross """
        print(f'Use dilated attn forward')

        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        l=h*w
        x_in = x

        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        
        dilate_slices = [
            make_dilate_index(x, offset=(0,0), flatten=True),
            make_dilate_index(x, offset=(0,1), flatten=True),
            make_dilate_index(x, offset=(1,0), flatten=True),
            make_dilate_index(x, offset=(1,1), flatten=True),
        ]
        if context[0] is not None:
            context = [c.repeat(4,1,1) for c in context] # repeat cond embedding to match the num_slices
        
        print(f'ori x ={x.shape}')
        
        # slicing & concat to batch
        # ------
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        # print(f'ori x flatten ={x.shape}')
        # x_full = x
        x = torch.cat([x[:, slice, :] for slice in dilate_slices], dim=0) # blc
        x=x.reshape(4,b,l//4,c).permute(1,0,2,3).contiguous().reshape(b*4,l//4,c)
        print(f'put slice to batch: {x.shape}') 
        # ------
        # pixel_unshuffle = nn.PixelUnshuffle(2)
        # x = pixel_unshuffle(x) # b,4c,h',w'
        # print(f'unshuffle={x.shape}')
        # x = rearrange(x, 'b (c n) h w -> (b n) c h w', n=4).contiguous() # stack in channel dim. ch * 4.
        # x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        # ------

        # forward x [4b, l//4, c]
        
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h//2, w=w//2).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        # x_in = rearrange(x_in, 'b (h w) c -> b c h w', h=h//2, w=w//2).contiguous()
        print(f'x={x.shape},x_in={x_in.shape}')

        # upsample from batch to h*w 
        # ------------------------------------------------
        # x = rearrange(x, '(b n) c h w -> b (c n) h w', n=4).contiguous() # stack in channel dim. ch * 4.
        # pixel_shuffle = nn.PixelShuffle(2)
        # print(f'x before unshuffle {x.shape}')
        # x = pixel_shuffle(x)
        # print(f'x afters unshuffle {x.shape}')
        # ------------------------------------------------
        # x = rearrange(x, '(b n) c h w -> b n c h w', n=4).contiguous() 
        # x = rearrange(x, 'b (nh nw) c h w -> b c h nh w nw', nh=2, nw=2).contiguous() 
        # x = rearrange(x, 'b c h nh w nw -> b c (h nh) (w nw)', nh=2, nw=2).contiguous() 
        # ------------------------------------------------
        x = x.contiguous().view(b, c, 2, 2, h//2, w//2)
        xout = x.permute(0, 1, 4, 2, 5, 3).contiguous() # b,c,h,2,w,2
        xout = xout.view(b, c, h, w)

        # x = xout

        x = xout + x_in

        
        return x