import string
import math
import numpy as np
import torch
import torch.nn.functional as F

from functools import partial
from torch import nn
from .normalizations import ConditionalInstanceNorm2dPlus


def get_act(config):
    if config.model.nonlinearity.lower() == 'elu':
        return nn.ELU()
    elif config.model.nonlinearity.lower() == 'relu':
        return nn.ReLU()
    elif config.model.nonlinearity.lower() == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif config.model.nonlinearity.lower() == 'swish':
        return nn.SiLU()
    else:
        raise NotImplementedError("activation function does not exits!")
    

def nscn_conv1x1(in_channels, out_channels, stride=1, bias=True, dilation=1, init_scale=1., padding=0):
    conv = nn.Conv2d(
        in_channels, out_channels, 
        kernel_size=1, stride=stride, padding=padding
        bias=bias, dilation=dilation=
    )
    init_scale = 1e-10 if init_scale == 0 else init_scale
    conv.weight.data *= init_scale
    conv.bias.data *= init_scale
    return conv


def variance_scaling(scale, mode, distribution, in_axis=1, out_axis=0, dtype=torch.float32, device='cpu'):
    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out
    
    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == 'fan_in':
            denominator = fan_in
        elif mode == 'fan_out':
            denominator = fan_out
        elif mode == 'fan_avg':
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(f"Invalid mode for variance scaling initializer: {mode}")
        
        variance = scale / denominator
        if distribution == 'normal':
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == 'uniform':
            return (torch.randn(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(variance)
        else:
            return ValueError("Invalid distribution for variance scaling initializer")
    
    return init


def default_init(scale=1.):
    """The same initialization used in DDPM"""
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, 'fan_avg', 'uniform')


class Dense(nn.Module):
    def __init__(self):
        super().__init__()
        
        
def ddpm_conv1x1(in_channels, out_channels, stride=1, bias=True, init_scale=1., padding=0):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, bias=bias)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


def ncsn_conv3x3(in_channels, out_channels, stride=1, bias=True, dilation=1, init_scale=1., padding=1):
    init_scale = 1e-10 if init_scale == 0 else init_scale
    conv = nn.Conv2d(in_channels, out_channels, stride=stride, bias=bias, dilation=dilation, padding=padding, kernel_size=3)
    conv.weight.data *= init_scale
    conv.bias.data *= init_scale
    return conv


def ddpm_conv3x3(in_channels, out_channels, stride=1, bias=True, dilation=1, init_scale=1., padding=1):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


class CRPBlock(nn.Moduel):
    def __init__(self, features, n_stages, act=nn.ReLU(), maxpool=True):
        super().__init__()
        self.convs == nn.ModuleList()
        for i in range(n_stages):
            self.convs.append(ncsn_conv3x3(features, features, stride=1, bias=False))
        self.n_stages = n_stages
        if maxpool:
            self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        else:
            self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        
        self.act = act
        
    def forward(self, x):
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            path = self.pool(path)
            path = self.convs[i](path)
            x = path + x
        return x
    

class CondCRPBlock(nn.Module):
    def __int__(self, features, n_stages, num_classes, normalizer, act=nn.ReLU()):
        super().__int__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_stages):
            self.norms.append(normalizer(features, num_classes, bias=True))
            self.convs.append(ncsn_conv3x3(features, features, stride=1, bias=False)) 
        self.n_stages = n_stages
        self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.act = act
        
    def forward(self, x, y):
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            path = self.norms[i](path, y)
            path = self.pool(path)
            path = self.convs[i](path)
            
            x = path + x
        return x
    

class RCUBlock(nn.Module):
    def __init__(self, features, n_blocks, n_stages, act=nn.ReLU()):
        super().__init__()
        
        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(self, f"{i+1}_{j+1}_conv", ncsn_conv3x3(features, features, stride=1, bias=False))
                
        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act
        
    def forward(self, x):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = self.act(x)
                x = getattr(self, f"{i+1}_{j+1}_conv")(x)
                
            x += residual
        return x   
    

class CondRCUBlock(nn.Module):
    def __init__(self, features, n_blocks, n_stages, num_classes, normalizer, act=nn.ReLU()):
        super().__init__()
        
        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(self, f"{i+1}_{j+1}_norm", normalizer(features, num_classes, bias=True))
                setattr(self, f"{i+1}_{j+1}_conv", ncsn_conv3x3(features, features, stride=1, bias=False))
        
        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act
        self.normalizer = normalizer
        
    def forward(self, x, y):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = getattr(self, f"{i+1}_{j+1}_norm")(x, y)
                x = self.act(x)
                x = getattr(self, f"{i+1}_{j+1}_conv")(z)
                
            x += residual
        return x
    
    
class MSFBlock(nn.Module):
    def __init__(self, in_channels, features):
        super().__init__()
        assert isinstance(in_channels, list) or isinstance(in_channels, tuple)
        self.convs = nn.ModuleList()
        self.features = features
        
        for i in range(len(in_channels)):
            self.convs.append(ncsn_conv3x3(in_channels[i], features, stride=1, bias=True))
            
    def forward(self, xs, shape):
        sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
        for i in range(len(self.convs)):
            h = self.convs[i](xs[i])
            h = F.interpolate(h, size=shape, mode='bilinear', align_corners=True)
            sums += h
        return sums
    
    
class CondMSFBlock(nn.Module):
    def __init__(self, in_channels, features, num_classes, normalizer):
        super().__init__()
        assert isinstance(in_channels, list) or isinstance(in_channels, tuple)
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.features = features
        self.normalizer = normalizer
        
        for i in range(len(in_channels)):
            self.convs.append(ncsn_conv3x3(in_channels[i], features, stride=1, bias=True))
            self.norms.append(normalizer(in_channels[i], num_classes, bias=True))
            
    def forward(self, xs, y, shape):
        sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
        for i in range(len(self.convs)):
            h = self.norms[i](xs[i], y)
            h = self.convs[i](h)
            h = F.interpolate(h, size=shape, mode='bilinear', align_corners=True)
            sums += h
        return sums
    

class RefineBlock(nn.Module):
    def __init__(self, in_channels, features, act=nn.ReLU(), start=False, end=False, maxpool=True):
        super().__init__()
        
        assert isinstance(in_channels, tuple) or isinstance(in_channels, list)
        self.n_blocks = n_blocks = len(in_channels)
        
        self.adapt_convs = nn.ModuleList()
        for i in range(n_blocks):
            self.adapt_convs.append(RCUBlock(in_channels[i], 2, 2, act))
            
        self.output_convs = RCUBlock(features, 3 if end else 1, 2, act)
        
        if not start:
            self.msf = MSFBlock(in_channels, features)
        
        self.crp = CRPBlock(features, 2, act, maxpool=maxpool)
        
    def forward(self, xs, output_shape):
        assert isinstance(xs, tuple) or isinstance(xs, list)
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i])
            hs.append(h)
            
        if self.n_blocks > 1:
            h = self.msf(hs, output_shape)
        else:
            h = hs[0]
        
        h = self.crp(h)
        h = self.output_convs(h)
        
        return h
    

class CondRefineBlock(nn.Module):
  def __init__(self, in_channels, features, num_classes, normalizer, act=nn.ReLU(), start=False, end=False):
    super().__init__()

    assert isinstance(in_channels, tuple) or isinstance(in_channels, list)
    self.n_blocks = n_blocks = len(in_channels)

    self.adapt_convs = nn.ModuleList()
    for i in range(n_blocks):
      self.adapt_convs.append(
        CondRCUBlock(in_channels[i], 2, 2, num_classes, normalizer, act)
      )

    self.output_convs = CondRCUBlock(features, 3 if end else 1, 2, num_classes, normalizer, act)

    if not start:
      self.msf = CondMSFBlock(in_channels, features, num_classes, normalizer)

    self.crp = CondCRPBlock(features, 2, num_classes, normalizer, act)

  def forward(self, xs, y, output_shape):
    assert isinstance(xs, tuple) or isinstance(xs, list)
    hs = []
    for i in range(len(xs)):
      h = self.adapt_convs[i](xs[i], y)
      hs.append(h)

    if self.n_blocks > 1:
      h = self.msf(hs, y, output_shape)
    else:
      h = hs[0]

    h = self.crp(h, y)
    h = self.output_convs(h, y)

    return h
        
        
class ConvMeanPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, biases=True, adjust_padding=False):
        super().__init__()
        if not adjust_padding:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
            self.conv = conv
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
            
            self.conv = nn.Sequential(
                nn.ZeroPad2d((1, 0, 1, 0)),
                conv
            )
    
    def forward(self, inputs):
        output = self.conv(inputs)
        output = sum(
            [output[:, :, ::2, ::2], output[:, :, 1::2, ::2],
             output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]] 
        ) / 3.
        return output
    

class MeanPoolConv(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, biases=True):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)

  def forward(self, inputs):
    output = inputs
    output = sum([output[:, :, ::2, ::2], output[:, :, 1::2, ::2],
                  output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    return self.conv(output)


class UpsampleConv(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, biases=True):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
    self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)

  def forward(self, inputs):
    output = inputs
    output = torch.cat([output, output, output, output], dim=1)
    output = self.pixelshuffle(output)
    return self.conv(output)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resample=None, act=nn.ELU(),
                 normalization=nn.InstanceNorm2d, adjust_padding=False, dilation=1):
        super().__init__()
        self.non_linearity = act
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resample = resample
        self.normalization = normalization
        if resample == 'down':
            if dilation > 1:
                self.conv1 = ncsn_conv3x3(in_channels, in_channels, dilation=dilation)
                self.norm2 = normalization(in_channels)
                self.conv2 = ncsn_conv3x3(in_channels, out_channels, dilation=dilation)
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
            else:
                self.conv1 = ncsn_conv3x3(in_channels, in_channels)
                self.norm2 = normalization(in_channels)
                self.conv2 = ConvMeanPool(in_channels, out_channels, 3, adjust_padding=adjust_padding)
                conv_shortcut = partial(ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding)
        elif resample is None:
            if dilation > 1:
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
                self.conv1 = ncsn_conv3x3(in_channels, out_channels, dilation=dilation)
                self.norm2 = normalization(out_channels)
                self.conv2 = ncsn_conv3x3(out_channels, out_channels, dilation=dilation)
            else:
                conv_shortcut = partial(nscn_conv1x1)
                self.conv1 = ncsn_conv3x3(in_channels, out_channels)
                self.norm2 = normalization(out_channels)
                self.conv2 = ncsn_conv3x3(out_channels, out_channels)
        else:
            raise Exception("Invalid resample value")
        
        if out_channels != in_channels or resample is not None:
            self.shortcut = conv_shortcut(in_channels, out_channels)
        
        self.norm1 = normalization(in_channels)
        
    def forward(self, x):
        output = self.norm1(x)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.norm2(output)
        output = self.non_linearity(output)
        output = self.conv2(output)
        
        if self.out_channels == self.in_channels and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)
        
        return shortcut + output
    
    
class ConditionalResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, resample=1, act=nn.ELU(),
                 normalization=ConditionalInstanceNorm2dPlus, adjust_padding=False, dilation=None):
        
        super().__init__()
        self.non_linearity = act
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resample = resample
        self.normalization = normalization
        if resample == 'down':
            if dilation > 1:
                self.conv1 = ncsn_conv3x3(in_channels, in_channels, dilation=dilation)
                self.norm2 = normalization(in_channels, num_classes)
                self.conv2 = ncsn_conv3x3(in_channels, out_channels, dilation=dilation)
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
            else:
                self.conv1 = ncsn_conv3x3(in_channels, in_channels)
                self.norm2 = normalization(in_channels, num_classes)
                self.conv2 = ConvMeanPool(in_channels, out_channels, 3, adjust_padding=adjust_padding)
                conv_shortcut = partial(ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding)
        elif resample is None:
            if dilation > 1:
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
                self.conv1 = ncsn_conv3x3(in_channels, out_channels, dilation=dilation)
                self.norm2 = normalization(out_channels, num_classes)
                self.conv2 = ncsn_conv3x3(out_channels, out_channels, dilation=dilation)
            else:
                conv_shortcut = partial(nscn_conv1x1)
                self.conv1 = ncsn_conv3x3(in_channels, out_channels)
                self.norm2 = normalization(out_channels, num_classes)
                self.conv2 = ncsn_conv3x3(out_channels, out_channels)
        else:
            raise Exception("Invalid resample value")
        
        if out_channels != in_channels or resample is not None:
            self.shortcut = conv_shortcut(in_channels, out_channels)
        
        self.norm1 = normalization(in_channels, num_classes)
        
    def forward(self, x, y):
        output = self.norm1(x, y)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.norm2(output, y)
        output = self.non_linearity(output)
        output = self.conv2(output)

        if self.out_channels == self.in_channels and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)

        return shortcut + output
    

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
  half_dim = embedding_dim // 2
  # magic number 10000 is from transformers
  emb = math.log(max_positions) / (half_dim - 1)
  # emb = math.log(2.) / (half_dim - 1)
  emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
  # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
  # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
  emb = timesteps.float()[:, None] * emb[None, :]
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = F.pad(emb, (0, 1), mode='constant')
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb


def _einsum(a, b, c, x, y):
  einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
  return torch.einsum(einsum_str, x, y)


def contract_inner(x, y):
  """tensordot(x, y, 1)."""
  x_chars = list(string.ascii_lowercase[:len(x.shape)])
  y_chars = list(string.ascii_lowercase[len(x.shape):len(y.shape) + len(x.shape)])
  y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
  out_chars = x_chars[:-1] + y_chars[1:]
  return _einsum(x_chars, y_chars, out_chars, x, y)


class NIN(nn.Module):
  def __init__(self, in_dim, num_units, init_scale=0.1):
    super().__init__()
    self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
    self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

  def forward(self, x):
    x = x.permute(0, 2, 3, 1)
    y = contract_inner(x, self.W) + self.b
    return y.permute(0, 3, 1, 2)


class AttentionBlock(nn.Module):
  """Channel-wise self-attention block."""
  def __init__(self, channels):
    super().__init__()
    self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
    self.nin_1 = NIN(channels, channels)
    self.nin_2 = NIN(channels, channels)
    self.nin_3 = NIN(channels, channels)
    self.nin_4 = NIN(channels, channels, init_scale=0.)

  def forward(self, x):
    B, C, H, W = x.shape
    h = self.norm(x)
    q = self.nin_1(h)
    k = self.nin_2(h)
    v = self.nin_3(h)

    w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
    w = torch.reshape(w, (B, H, W, H * W))
    w = F.softmax(w, dim=-1)
    w = torch.reshape(w, (B, H, W, H, W))
    h = torch.einsum('bhwij,bcij->bchw', w, v)
    h = self.nin_4(h)
    return x + h


class Upsample(nn.Module):
  def __init__(self, channels, with_conv=False):
    super().__init__()
    if with_conv:
      self.conv = ddpm_conv3x3(channels, channels)
    self.with_conv = with_conv

  def forward(self, x):
    B, C, H, W = x.shape
    h = F.interpolate(x, (H * 2, W * 2), mode='nearest')
    if self.with_conv:
      h = self.conv(h)
    return h


class Downsample(nn.Module):
  def __init__(self, channels, with_conv=False):
    super().__init__()
    if with_conv:
      self.conv = ddpm_conv3x3(channels, channels, stride=2, padding=0)
    self.with_conv = with_conv

  def forward(self, x):
    B, C, H, W = x.shape
    # Emulate 'SAME' padding
    if self.with_conv:
      x = F.pad(x, (0, 1, 0, 1))
      x = self.conv(x)
    else:
      x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)

    assert x.shape == (B, C, H // 2, W // 2)
    return x


class ResnetBlockDDPM(nn.Module):
    def __init__(self, act, in_channels, out_channels=None, temb_dim=None, conv_shortcut=False, dropout=0.1):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.act = act
        self.conv1 = ddpm_conv3x3(in_channels, out_channels)
        if temb_dim is not None:
            self.dense1 = nn.Linear(temb_dim, out_channels)
            self.dense1.weight.data = default_init()(self.dense1.weight.data.shape)
            nn.init.zeros_(self.dense1.bias)
        
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = ddpm_conv3x3(out_channels, out_channels, init_scale=0.)
        if in_channels != out_channels:
            if conv_shortcut:
                self.conv3 = ddpm_conv3x3(in_channels, out_channels)
            else:
                self.nin1 = NIN(in_channels, out_channels)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.conv_shortcut = conv_shortcut
        
    def forward(self, x, temb=None):
        B, C, H, W = x.shape
        assert C == self.in_channels
        out_channels = self.out_channels if self.out_channels else self.in_channels
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        if temb is not None:
            h += self.dense1(self.act(temb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.dropout1(h)
        h = self.conv2(h)
        if C != out_channels:
            if self.conv_shortcut:
                x = self.conv3(x)
            else:
                x = self.nin1(x)
        return x + h