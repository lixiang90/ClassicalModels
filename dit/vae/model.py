import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        x = self.nin_shortcut(x)
        
        return x + h

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
        else:
            self.conv = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if hasattr(self, 'conv') and isinstance(self.conv, nn.Conv2d):
            # Asymmetric padding for standard torch conv behavior with stride 2
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = self.conv(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels=3, ch=128, ch_mult=(1, 2, 4, 4), num_res_blocks=2, z_channels=4, dropout=0.0):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        
        curr_res = 256 # Just for tracking, assumes 256 input
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        
        self.down = nn.ModuleList()
        block_in = ch
        
        # Downsampling loop
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                
            down = nn.Module()
            down.block = block
            
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, with_conv=True)
                curr_res = curr_res // 2
                
            self.down.append(down)
            
        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        # self.mid.attn_1 = AttnBlock(block_in) # Optional
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        
        # End
        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        # x: [B, 3, 256, 256]
        hs = [self.conv_in(x)]
        
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                hs.append(h)
                
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
                
        h = hs[-1]
        h = self.mid.block_1(h)
        # h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h

class Decoder(nn.Module):
    def __init__(self, out_channels=3, ch=128, ch_mult=(1, 2, 4, 4), num_res_blocks=2, z_channels=4, dropout=0.0):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = 256 // 2**(self.num_resolutions - 1)
        
        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        
        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        # self.mid.attn_1 = AttnBlock(block_in) # Optional
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        
        # Upsampling loop
        self.up = nn.ModuleList()
        
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                
            up = nn.Module()
            up.block = block
            
            if i_level != 0:
                up.upsample = Upsample(block_in, with_conv=True)
                curr_res = curr_res * 2
                
            self.up.append(up)
            
        # End
        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, z):
        # z: [B, 4, 32, 32]
        h = self.conv_in(z)
        
        h = self.mid.block_1(h)
        # h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
            
            if i_level != self.num_resolutions - 1:
                h = self.up[i_level].upsample(h)
                
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h

class VAE(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 z_channels=4, 
                 ch=128, 
                 ch_mult=(1, 2, 4, 4), # 3 layers of downsampling: 256->128->64->32
                 num_res_blocks=2, 
                 dropout=0.0):
        super().__init__()
        
        self.encoder = Encoder(in_channels=in_channels, ch=ch, ch_mult=ch_mult, 
                               num_res_blocks=num_res_blocks, z_channels=z_channels, dropout=dropout)
        self.decoder = Decoder(out_channels=in_channels, ch=ch, ch_mult=ch_mult, 
                               num_res_blocks=num_res_blocks, z_channels=z_channels, dropout=dropout)
        self.conv_quant = nn.Conv2d(2*z_channels, 2*z_channels, 1)
        self.conv_post = nn.Conv2d(z_channels, z_channels, 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.conv_quant(h)
        posterior = DiagonalGaussianDistribution(h)
        return posterior

    def decode(self, z):
        z = self.conv_post(z)
        dec = self.decoder(z)
        return dec

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn_like(self.mean).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        if other is None:
            return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                   + self.var - 1.0 - self.logvar,
                                   dim=[1, 2, 3])
        else:
            return 0.5 * torch.sum(
                torch.pow(self.mean - other.mean, 2) / other.var
                + self.var / other.var - 1.0 - self.logvar + other.logvar,
                dim=[1, 2, 3])

    def mode(self):
        return self.mean

if __name__ == "__main__":
    # Simple test
    model = VAE(in_channels=3, z_channels=4, ch=64, ch_mult=(1, 2, 4, 4))
    x = torch.randn(1, 3, 256, 256)
    dec, posterior = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Latent mean shape: {posterior.mean.shape}")
    print(f"Output shape: {dec.shape}")
    
    assert posterior.mean.shape == (1, 4, 32, 32)
    assert dec.shape == (1, 3, 256, 256)
    print("Test passed!")
