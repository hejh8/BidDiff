import torch
import torch.nn as nn
import torch.nn.functional as F

class RetinexDecomposition(nn.Module):
    def __init__(self, in_channels=3, num_features=32):
        super().__init__()
       
        self.illumination_net = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(num_features, 1, 3, padding=1, padding_mode='reflect'),
            nn.Sigmoid()  
        )
    
    def forward(self, I_low):
        L = self.illumination_net(I_low)
        R = I_low / (L + 1e-6)  
        return L, R


class ReflectionGuidedAttention(nn.Module):
    
    def __init__(self, channels):
        super().__init__()
        
        self.r_conv = nn.Sequential(
            nn.Conv2d(3, channels, 1),
            nn.ReLU()
        )
        
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//8, 1),
            nn.ReLU(),
            nn.Conv2d(channels//8, channels, 1),
            nn.Sigmoid()
        )
        
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect'),
            nn.Sigmoid()
        )
    
    def forward(self, x, R):
        # 反射图特征
        r_feat = self.r_conv(R)
        
        # 通道注意力
        ca = self.channel_att(x + r_feat)  # 反射图引导通道注意力
        
        # 空间注意力
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa = self.spatial_att(torch.cat([max_pool, avg_pool], dim=1))
        
        return x * ca * sa  


class RetinexColorCorrectionNet(nn.Module):
    
    def __init__(self, in_channels=3, num_features=32):
        super().__init__()
        
        self.retinex = RetinexDecomposition(in_channels, num_features)
        
    
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features, 3, padding=1, padding_mode='reflect'),
            nn.ReLU()
        )
        
        
        self.attention = ReflectionGuidedAttention(num_features)
        
       
        self.color_correction = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(num_features, in_channels, 3, padding=1, padding_mode='reflect'),
            nn.Sigmoid()  
        )
    
    def forward(self, I_low):

        L, R = self.retinex(I_low)
        features = self.feature_extractor(I_low)
        enhanced_features = self.attention(features, R)

        I_corrected = self.color_correction(enhanced_features)
        return I_corrected