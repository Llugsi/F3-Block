# ===================================================================================================
# BLIND-SPOT CONSTRAINT VALIDATION & HERMETICITY PROPERTY - MULTI-LAYER RECONSTRUCTION
# ===================================================================================================
# THEORETICAL ASSURANCE: This reconstruction block strictly adheres to the 
# blind-spot theorem. Because the four independent causal streams (feat_N, feat_S, feat_E, feat_W) 
# were physically deprived of the central coordinate (y, x) via pure asymmetric truncation, the 
# mutual information between the original input pixel and the combined tensor 'fused_features' at 
# that specific coordinate is strictly equal to zero. Since the subsequent projection relies exclusively 
# on convolutional operators with unit local geometry (kernel_size=1) and element-wise activation 
# functions (LeakyReLU), the mathematical transformation operates solely across the channel dimension. 
# In the absence of spatial expansion or coupling with adjacent coordinates, it is algebraically 
# impossible for the non-linearity to decode, map, or recover the pristine central input value. 
# The feature space remains hermetically sealed, entirely eliminating the risk of cross-branch data 
# leakage and forcing the optimization loop to converge solely toward the geological structural coherence 
# of subsurface reflectors under the self-supervised Noise2Noise paradigm.
# ===================================================================================================

# [Script 3] Pure Causal Conv2D operators and BlindSpotNet architecture
class TNNLS_CausalConv2d(nn.Module):
    """
    Pure Directional Causal Convolutional Operator.
    Guarantees an absolute blind-spot of size R + 1 without data leakage.
    """
    def __init__(self, direction, in_ch, out_ch, kernel_size=3, dilation=2):
        super(TNNLS_CausalConv2d, self).__init__()
        self.direction = direction
        
        self.R = (kernel_size - 1) * dilation // 2 
        self.shift = self.R + 1 
        
        if direction in ['N', 'S']:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size, 1), 
                                  padding=0, dilation=(dilation, 1))
        elif direction in ['E', 'W']:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(1, kernel_size), 
                                  padding=0, dilation=(1, dilation))

    def forward(self, x):
        if self.direction == 'N':
            x_padded = F.pad(x, (0, 0, self.R, self.R + self.shift))
            out = self.conv(x_padded)
            return out[:, :, self.shift:, :]
            
        elif self.direction == 'S':
            x_padded = F.pad(x, (0, 0, self.R + self.shift, self.R))
            out = self.conv(x_padded)
            return out[:, :, :-self.shift, :]
            
        elif self.direction == 'E':
            x_padded = F.pad(x, (self.R + self.shift, self.R, 0, 0))
            out = self.conv(x_padded)
            return out[:, :, :, :-self.shift]
            
        elif self.direction == 'W':
            x_padded = F.pad(x, (self.R, self.R + self.shift, 0, 0))
            out = self.conv(x_padded)
            return out[:, :, :, self.shift:]


class TNNLS_BlindSpotNet(nn.Module):
    """
    Corrected Anisotropic Dilated Causal Blind-Spot Network.
    Centralized cross-attention gating was removed to prevent cross-branch data leakage 
    and maintain the absolute integrity of the spatial blind-spot constraint.
    """
    def __init__(self, in_channels=1, out_channels=1, features=32):
        super(TNNLS_BlindSpotNet, self).__init__()
        
        # Pure causal branches with guaranteed blind-spot constraints
        self.branch_N = nn.Sequential(
            TNNLS_CausalConv2d('N', in_channels, features, kernel_size=3, dilation=1),
            nn.LeakyReLU(0.1),
            TNNLS_CausalConv2d('N', features, features, kernel_size=3, dilation=2),
            nn.LeakyReLU(0.1)
        )
        
        self.branch_S = nn.Sequential(
            TNNLS_CausalConv2d('S', in_channels, features, kernel_size=3, dilation=1),
            nn.LeakyReLU(0.1),
            TNNLS_CausalConv2d('S', features, features, kernel_size=3, dilation=2),
            nn.LeakyReLU(0.1)
        )
        
        self.branch_E = nn.Sequential(
            TNNLS_CausalConv2d('E', in_channels, features, kernel_size=3, dilation=1),
            nn.LeakyReLU(0.1),
            TNNLS_CausalConv2d('E', features, features, kernel_size=3, dilation=2),
            nn.LeakyReLU(0.1)
        )
        
        self.branch_W = nn.Sequential(
            TNNLS_CausalConv2d('W', in_channels, features, kernel_size=3, dilation=1),
            nn.LeakyReLU(0.1),
            TNNLS_CausalConv2d('W', features, features, kernel_size=3, dilation=2),
            nn.LeakyReLU(0.1)
        )
        
        # Branch-isolated relevance score generators (1x1 Convolutions)
        # This isolates features across streams during the attention weighting phase
        self.attn_map_N = nn.Conv2d(features, 1, kernel_size=1)
        self.attn_map_S = nn.Conv2d(features, 1, kernel_size=1)
        self.attn_map_E = nn.Conv2d(features, 1, kernel_size=1)
        self.attn_map_W = nn.Conv2d(features, 1, kernel_size=1)
        
        # Final reconstruction using 1x1 convolutions
        self.reconstruction = nn.Sequential(
            nn.Conv2d(features, 32, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # 1. Strict causal feature extraction
        feat_N = self.branch_N(x)
        feat_S = self.branch_S(x)
        feat_E = self.branch_E(x)
        feat_W = self.branch_W(x)
        
        # 2. Score mapping within isolated branches
        score_N = self.attn_map_N(feat_N)
        score_S = self.attn_map_S(feat_S)
        score_E = self.attn_map_E(feat_E)
        score_W = self.attn_map_W(feat_W)
        
        # 3. Concatenate scalar scores to apply channel-wise normalization
        scores = torch.cat([score_N, score_S, score_E, score_W], dim=1)
        attn_weights = F.softmax(scores, dim=1)
        
        # 4. Independent weight split per direction
        w_N, w_S, w_E, w_W = torch.chunk(attn_weights, chunks=4, dim=1)
        
        # 5. Mathematically secure anisotropic weighted fusion
        fused_features = (feat_N * w_N) + (feat_S * w_S) + (feat_E * w_E) + (feat_W * w_W)
        
        return self.reconstruction(fused_features)
