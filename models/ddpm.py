import torch
import functools

from torch import nn
from . import utils, normalizations, layers
from .normalizations import get_normalization
from .layers import (
    RefineBlock,
    ResidualBlock,
    ResnetBlockDDPM,
    Upsample,
    Downsample,
    ddpm_conv3x3,
    default_init,
    get_act
)

@utils.register_model(name='ddpm')
class DDPM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.act = act = get_act(config)
        self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))
        
        self.num_features = num_features = config.model.nf
        channel_multiplier = config.model.ch_mult
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        resample_with_conv = config.model.resamp_with_conv
        self.num_resolutions = num_resolutions = len(channel_multiplier)
        self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]
        
        AttentionBlock = functools.partial(layers.AttentionBlock)
        self.conditional = conditional = config.model.conditional
        ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * num_features, dropout=dropout)
        if conditional:
            modules = [nn.Linear(num_features, num_features * 4)]
            modules[0].weight.data = default_init()(modules[0].weight.data.shape)
            nn.init.zeros_(modules[0].bias)
            modules.append(nn.Linear(num_features * 4 , num_features * 4))
            modules[1].weight.data = default_init()(modules[1].weight.data.shape)
            nn.init.zeros_(modules[1].bias)
            
        self.centered = config.data.centered
        channels = config.data.num_channels
        
        # Downsampling Layers
        modules.append(ddpm_conv3x3(channels, num_features))
        hs_c = [num_features]
        in_channels = num_features
        for i in range(num_resolutions):
            for j in range(num_res_blocks):
                out_channels = num_features * channel_multiplier[i]
                modules.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if all_resolutions[i] in attn_resolutions:
                    modules.append(AttentionBlock(channels=in_channels))
                hs_c.append(in_channels)
            if i != num_resolutions - 1:
                modules.append(Downsample(in_channels, with_conv=resample_with_conv))
                hs_c.append(in_channels)
                
        in_channels = hs_c[-1]
        modules.append(ResnetBlock(in_channels=in_channels))
        modules.append(AttentionBlock(channels=in_channels))
        modules.append(ResnetBlock(in_channels=in_channels))
        
        # Upsampling Layers
        for i in reversed(range(num_resolutions)):
            for j in range(num_res_blocks + 1):
                out_channels = num_features * channel_multiplier[i]
                modules.append(ResnetBlock(in_channels + hs_c.pop(), out_channels))
                in_channels = out_channels
            if all_resolutions[i] in attn_resolutions:
                modules.append(AttentionBlock(channels=in_channels))
            if i != 0:
                modules.append(Upsample(in_channels, with_conv=resample_with_conv))
            
        assert not hs_c
        modules.append(nn.GroupNorm(num_channels=in_channels, num_groups=32, eps=1e-6))
        modules.append(ddpm_conv3x3(in_channels, channels, init_scale=0.))
        self.modules = nn.ModuleList(modules)
        
        self.scale_by_sigma = config.model.scale_by_sigma
        
    def forward(self, x, labels):
        modules = self.modules
        m_idx = 0
        if self.conditional:
            timesteps = labels
            temb = layers.get_timestep_embedding(timesteps, self.num_features)
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx]((self.act(temb)))
            m_idx += 1
        else:
            temb = None
            
        if self.ceneter:
            # Input is in [-1, 1]
            h = x
        else:
            # Input is in [0, 1]
            h = 2 * x - 1.
            
        # Downsampling
        hs = [modules[m_idx](h)]
        m_idx += 1
        for i in range(self.num_resolutions):
            for j in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)
            if i != self.num_resolutions - 1:
                hs.append(modules[m_idx](hs[-1]))
                m_idx += 1
                
        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1
        
        # Upsampling
        for i in reversed(range(self.num_resolutions)):
            for j in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1
            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1
            if i != 0 :
                h = modules[m_idx](h)
                m_idx += 1
                
        assert not hs
        h = self.act(modules[m_idx](h))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        assert m_idx == len(modules)
        
        if self.scale_by_sigma:
            used_sigmas = self.sigmas[labels, None, None, None]
            h = h / used_sigmas
            
        return h 