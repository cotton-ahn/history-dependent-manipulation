#Codes based on : https://github.com/princeton-vl/pytorch_stacked_hourglass/blob/master/models/layers.py
    
import torch
from torch import nn
import math
import numpy as np
from utils import generate_spatial_batch
import torch.nn.functional as F

bs = 1
Pool = nn.MaxPool2d

def batchnorm(x):
    return nn.BatchNorm2d(x.size()[1])(x)

img_size = 256
spatial_coords = [generate_spatial_batch(bs, 2**i, 2**i) for i in range(2, 8)]

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
        
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out 

class Hourglass_history(nn.Module):
    def __init__(self, n, f, hist_temp, bn=None, increase=0):
        super(Hourglass_history, self).__init__()
        self.hist_temp = hist_temp
        nf = f + increase
        self.up1 = Residual(f, f) # f+8
        # Lower branch
        self.pool1 = Pool(2, 2)
        self.pool2 = Pool(4, 4)
        self.low1 = Residual(f, nf)
        self.n = n
        self.f = f
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass_history(n-1, nf, hist_temp, bn=bn)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf*3, f) # prev nf*3
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, word_embed, histories, time, history_rnn, W_w, W_h, Ww_q, Ww_k, Ww_v, Wh_q, Wh_k, Wh_v, device='cuda'):  
        B, D, H, W = x.shape

        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        if self.n > 1:
            low2, curr_feat = self.low2(low1, word_embed, histories, time, history_rnn, W_w, W_h, Ww_q, Ww_k, Ww_v, Wh_q, Wh_k, Wh_v)
            self.curr_feat = curr_feat
        else:
            low2 = self.low2(low1)
        
        B, D, H, W = low2.shape
        new_low2 = list()
        for i, spa_feat in enumerate(low2):
            spa_feat = spa_feat.view(D, H*W).permute(1, 0) # HW D
            lang_feat = word_embed[i].squeeze() # T, D
            
            spa_query = Ww_q[self.n-1](spa_feat)
            lang_key = Ww_k[self.n-1](lang_feat)
            lang_value = Ww_v[self.n-1](lang_feat)
                        
            attn = torch.mm(spa_query, lang_key.permute(1, 0))
            attn = torch.softmax(attn/(self.hist_temp*math.sqrt(D)), dim=-1)
            attn_val = torch.mm(attn, lang_value)
            
            fuse_feat = torch.cat([spa_feat, attn_val], dim=-1).view(H, W, 2*D).permute(2, 0, 1)
            new_low2.append(fuse_feat[None, :, :, :])
            
        low2 = torch.cat(new_low2, dim=0)

        del new_low2
        torch.cuda.empty_cache()
        
        B, D, H, W = low2.shape
        
        ### NEW
        
        # num_layers = 2
        if self.n == 1: 
            self.curr_feat = torch.mean(low2, dim=[2, 3]).view(low2.shape[0], low2.shape[1]) # B X D 
        
        if len(histories) == 0:
            hist_feat = torch.zeros(B, (D)//2, H, W).to(device)
            low2 = torch.cat([low2, hist_feat], dim=1)
        else:
            new_low2 = list()
            histories = torch.cat(histories, dim=1) # B, T, D
            histories, _ = history_rnn(histories)
            
            for i, spa_feat in enumerate(low2):
                curr_time_feat = time[i][None, :].repeat(H*W, 1)
                spa_feat = spa_feat.view(D, H*W).permute(1, 0) # spa_feat HW D
                
                spa_query = Wh_q[self.n-1](spa_feat) # # spafeattime->spafeatHW D
                hist_key = Wh_k[self.n-1](histories[i]) # T D
                hist_value = Wh_v[self.n-1](histories[i])
                attn = torch.mm(spa_query, hist_key.permute(1, 0))
                attn = torch.softmax(attn/(self.hist_temp*math.sqrt(D)), dim=-1)
                attn_val = torch.mm(attn, hist_value)
                
                fuse_feat = torch.cat([spa_feat, attn_val], dim=-1).view(H, W, 3*((D)//2)).permute(2, 0, 1)
                new_low2.append(fuse_feat[None, :, :, :])
                
            low2 = torch.cat(new_low2, dim=0)
            del new_low2
            torch.cuda.empty_cache()    
                
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        
        return up1 + up2, self.curr_feat
    
    
class Hourglass_nonhistory(nn.Module):
    def __init__(self, n, f, hist_temp, bn=None, increase=0):
        super(Hourglass_nonhistory, self).__init__()
        self.hist_temp = hist_temp
        nf = f + increase
        self.up1 = Residual(f, f) # f+8
        # Lower branch
        self.pool1 = Pool(2, 2)
        self.pool2 = Pool(4, 4)
        self.low1 = Residual(f, nf)
        self.n = n
        self.f = f
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass_nonhistory(n-1, nf, hist_temp, bn=bn)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf*2, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, word_embed, Ww_q, Ww_k, Ww_v, device='cuda'):  
        B = x.shape[0]
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        if self.n > 1:
            low2 = self.low2(low1, word_embed, Ww_q, Ww_k, Ww_v)
        else:
            low2 = self.low2(low1)
        
        B, D, H, W = low2.shape
        new_low2 = list()
        for i, spa_feat in enumerate(low2):
            spa_feat = spa_feat.view(D, H*W).permute(1, 0) # HW D
            lang_feat = word_embed[i].squeeze() # T, D
            
            spa_query = Ww_q[self.n-1](spa_feat)
            lang_key = Ww_k[self.n-1](lang_feat)
            lang_value = Ww_v[self.n-1](lang_feat)
            
            attn = torch.mm(spa_query, lang_key.permute(1, 0))
            attn = torch.softmax(attn/(self.hist_temp*math.sqrt(D)), dim=-1)
            attn_val = torch.mm(attn, lang_value)
            
            fuse_feat = torch.cat([spa_feat, attn_val], dim=-1).view(H, W, 2*D).permute(2, 0, 1)
            new_low2.append(fuse_feat[None, :, :, :])
            
        low2 = torch.cat(new_low2, dim=0)
        del new_low2
        torch.cuda.empty_cache() 
                
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        
        return up1 + up2