import math
import torch
import torch.nn as nn
import torch.nn.functional

# This script is from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
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
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_




class DualDiffusionUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ch = config.model.ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(config.model.ch_mult)
        self.num_res_blocks = config.model.num_res_blocks
        self.in_channels = config.model.in_channels * 2  # 输入为低光和高光图像的拼接

        
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            nn.Linear(self.ch, self.temb_ch),
            nn.Linear(self.temb_ch, self.temb_ch),
        ])

       
        self.conv_in = nn.Conv2d(self.in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        self.down = nn.ModuleList()  

        
        self.bidirectional_attn = nn.ModuleList()  

        
        in_ch_mult = (1,) + tuple(config.model.ch_mult)
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = self.ch * in_ch_mult[i_level]
            block_out = self.ch * config.model.ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         temb_channels=self.temb_ch, dropout=config.model.dropout))
                block_in = block_out
                if i_level == 2:  
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, config.model.resamp_with_conv)
            self.down.append(down)

            
            if i_level > 0:  
                self.bidirectional_attn.append(BidirectionalAttention(block_in))

        
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=config.model.dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=config.model.dropout)

        
        self.up_low2high = nn.ModuleList()  
        self.up_high2low = nn.ModuleList()  
        for i_level in reversed(range(self.num_resolutions)):
            block_low2high = nn.ModuleList()
            block_high2low = nn.ModuleList()
            attn_low2high = nn.ModuleList()
            attn_high2low = nn.ModuleList()

            block_out = self.ch * config.model.ch_mult[i_level]
            skip_in = self.ch * config.model.ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = self.ch * in_ch_mult[i_level]
                
                block_low2high.append(ResnetBlock(in_channels=block_in + skip_in, out_channels=block_out,
                                                 temb_channels=self.temb_ch, dropout=config.model.dropout))
               
                block_high2low.append(ResnetBlock(in_channels=block_in + skip_in, out_channels=block_out,
                                                 temb_channels=self.temb_ch, dropout=config.model.dropout))
                block_in = block_out
                if i_level == 2:  
                    attn_low2high.append(AttnBlock(block_in))
                    attn_high2low.append(AttnBlock(block_in))

            up_low2high = nn.Module()
            up_low2high.block = block_low2high
            up_low2high.attn = attn_low2high
            if i_level != 0:
                up_low2high.upsample = Upsample(block_in, config.model.resamp_with_conv)

            up_high2low = nn.Module()
            up_high2low.block = block_high2low
            up_high2low.attn = attn_high2low
            if i_level != 0:
                up_high2low.upsample = Upsample(block_in, config.model.resamp_with_conv)

            self.up_low2high.insert(0, up_low2high)
            self.up_high2low.insert(0, up_high2low)

        
        self.norm_out = Normalize(block_in)
        self.conv_out_low2high = nn.Conv2d(block_in, config.model.out_ch, kernel_size=3, stride=1, padding=1)
        self.conv_out_high2low = nn.Conv2d(block_in, config.model.out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t, direction='low2high'):

    
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

       
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

              
                if i_level > 0:
                    h_low2high, h_high2low = self.bidirectional_attn[i_level - 1](hs[-1], temb)
                    hs[-1] = h_low2high if direction == 'low2high' else h_high2low

        
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

       
        if direction == 'low2high':
            up_blocks = self.up_low2high
            conv_out = self.conv_out_low2high
        else:
            up_blocks = self.up_high2low
            conv_out = self.conv_out_high2low

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = up_blocks[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)
                if len(up_blocks[i_level].attn) > 0:
                    h = up_blocks[i_level].attn[i_block](h)
            if i_level != 0:
                h = up_blocks[i_level].upsample(h)

       
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = conv_out(h)
        return h


class DualDiffusionUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ch = config.model.ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(config.model.ch_mult)
        self.num_res_blocks = config.model.num_res_blocks
        self.in_channels = config.model.in_channels * 2  # 输入为低光和高光图像的拼接

       
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            nn.Linear(self.ch, self.temb_ch),
            nn.Linear(self.temb_ch, self.temb_ch),
        ])

        
        self.conv_in = nn.Conv2d(self.in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        self.down = nn.ModuleList()  

       
        self.bidirectional_attn = nn.ModuleList()  

        
        in_ch_mult = (1,) + tuple(config.model.ch_mult)
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = self.ch * in_ch_mult[i_level]
            block_out = self.ch * config.model.ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         temb_channels=self.temb_ch, dropout=config.model.dropout))
                block_in = block_out
                if i_level == 2:  
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, config.model.resamp_with_conv)
            self.down.append(down)

            
            if i_level > 0:  
                self.bidirectional_attn.append(BidirectionalAttention(block_in))

        
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=config.model.dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=config.model.dropout)

       
        self.up_low2high = nn.ModuleList() 
        self.up_high2low = nn.ModuleList()  
        for i_level in reversed(range(self.num_resolutions)):
            block_low2high = nn.ModuleList()
            block_high2low = nn.ModuleList()
            attn_low2high = nn.ModuleList()
            attn_high2low = nn.ModuleList()

            block_out = self.ch * config.model.ch_mult[i_level]
            skip_in = self.ch * config.model.ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = self.ch * in_ch_mult[i_level]
               
                block_low2high.append(ResnetBlock(in_channels=block_in + skip_in, out_channels=block_out,
                                                 temb_channels=self.temb_ch, dropout=config.model.dropout))
               
                block_high2low.append(ResnetBlock(in_channels=block_in + skip_in, out_channels=block_out,
                                                 temb_channels=self.temb_ch, dropout=config.model.dropout))
                block_in = block_out
                if i_level == 2: 
                    attn_low2high.append(AttnBlock(block_in))
                    attn_high2low.append(AttnBlock(block_in))

            up_low2high = nn.Module()
            up_low2high.block = block_low2high
            up_low2high.attn = attn_low2high
            if i_level != 0:
                up_low2high.upsample = Upsample(block_in, config.model.resamp_with_conv)

            up_high2low = nn.Module()
            up_high2low.block = block_high2low
            up_high2low.attn = attn_high2low
            if i_level != 0:
                up_high2low.upsample = Upsample(block_in, config.model.resamp_with_conv)

            self.up_low2high.insert(0, up_low2high)
            self.up_high2low.insert(0, up_high2low)

        
        self.norm_out = Normalize(block_in)
        self.conv_out_low2high = nn.Conv2d(block_in, config.model.out_ch, kernel_size=3, stride=1, padding=1)
        self.conv_out_high2low = nn.Conv2d(block_in, config.model.out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t, direction='low2high'):

        
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

       
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

                
                if i_level > 0:
                    h_low2high, h_high2low = self.bidirectional_attn[i_level - 1](hs[-1], temb)
                    hs[-1] = h_low2high if direction == 'low2high' else h_high2low

        
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        
        if direction == 'low2high':
            up_blocks = self.up_low2high
            conv_out = self.conv_out_low2high
        else:
            up_blocks = self.up_high2low
            conv_out = self.conv_out_high2low

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = up_blocks[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)
                if len(up_blocks[i_level].attn) > 0:
                    h = up_blocks[i_level].attn[i_block](h)
            if i_level != 0:
                h = up_blocks[i_level].upsample(h)

        
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = conv_out(h)
        return h


class BidirectionalAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)  
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  

    def forward(self, x, temb):
        batch_size, _, height, width = x.shape

        
        q_low2high = self.query(x)  # (batch_size, channels//8, height, width)
        k_low2high = self.key(x)    # (batch_size, channels//8, height, width)
        v_low2high = self.value(x)  # (batch_size, channels, height, width)

        
        q_low2high = q_low2high.view(batch_size, -1, height * width)  # (batch_size, channels//8, height*width)
        k_low2high = k_low2high.view(batch_size, -1, height * width)  # (batch_size, channels//8, height*width)
        v_low2high = v_low2high.view(batch_size, -1, height * width)  # (batch_size, channels, height*width)

        
        attn_low2high = torch.matmul(q_low2high.transpose(1, 2), k_low2high)  # (batch_size, height*width, height*width)
        attn_low2high = torch.softmax(attn_low2high, dim=-1)

        
        out_low2high = torch.matmul(v_low2high, attn_low2high.transpose(1, 2))  # (batch_size, channels, height*width)
        out_low2high = out_low2high.view(batch_size, -1, height, width)  # 恢复空间维度

        
        q_high2low = self.query(x)  # (batch_size, channels//8, height, width)
        k_high2low = self.key(x)    # (batch_size, channels//8, height, width)
        v_high2low = self.value(x)  # (batch_size, channels, height, width)

        
        q_high2low = q_high2low.view(batch_size, -1, height * width)  # (batch_size, channels//8, height*width)
        k_high2low = k_high2low.view(batch_size, -1, height * width)  # (batch_size, channels//8, height*width)
        v_high2low = v_high2low.view(batch_size, -1, height * width)  # (batch_size, channels, height*width)

        
        attn_high2low = torch.matmul(q_high2low.transpose(1, 2), k_high2low)  # (batch_size, height*width, height*width)
        attn_high2low = torch.softmax(attn_high2low, dim=-1)

       
        out_high2low = torch.matmul(v_high2low, attn_high2low.transpose(1, 2))  # (batch_size, channels, height*width)
        out_high2low = out_high2low.view(batch_size, -1, height, width) 

        
        out_low2high = x + self.gamma * out_low2high
        out_high2low = x + self.gamma * out_high2low

        return out_low2high, out_high2low