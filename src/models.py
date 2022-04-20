import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Conv, Hourglass_history, Pool, Residual, Hourglass_nonhistory

bs = 1
img_size = 256
sc_dim = 2

class IT2P_history(nn.Module):
    def __init__(self, inp_dim, oup_dim, dictionary, word_embed_dim, temp, time_embed_dim=64, depth=4, t_length=8, bn=False, increase=0):
        super(IT2P_history, self).__init__()
        
        self.time_embedder = nn.Embedding(t_length, time_embed_dim)
                
        self.word_embed_dim = word_embed_dim
        self.word_embedder = nn.Embedding(len(dictionary), word_embed_dim)
        
        self.word_rnn = nn.LSTM(word_embed_dim, word_embed_dim, batch_first=True, num_layers=1, bidirectional=True)
        #inp_dim//2 => inp_dim
        self.history_rnn = nn.LSTM(inp_dim*2+time_embed_dim, inp_dim, batch_first=True, num_layers=1, bidirectional=True)
        
        self.W_w = nn.ModuleList([nn.Linear(word_embed_dim*2, (img_size//2**(depth+3-i))**2, bias=False) for i in range(1, depth+1)])
        self.W_h = nn.ModuleList([nn.Linear(inp_dim, (img_size//2**(depth+3-i))**2, bias=False) for i in range(1, depth+1)])
        
        # set bias False
        self.Ww_q = nn.ModuleList([nn.Linear(inp_dim, inp_dim, bias=False) for i in range(1, depth+1)])
        self.Ww_k = nn.ModuleList([nn.Linear(word_embed_dim*2, inp_dim, bias=False) for i in range(1, depth+1)])
        self.Ww_v = nn.ModuleList([nn.Linear(word_embed_dim*2, inp_dim, bias=False) for i in range(1, depth+1)])
        
        # erase time_emb_dim
        self.Wh_q = nn.ModuleList([nn.Linear(inp_dim*2, inp_dim, bias=False) for i in range(1, depth+1)])
        self.Wh_k = nn.ModuleList([nn.Linear(inp_dim*2, inp_dim, bias=False) for i in range(1, depth+1)])
        self.Wh_v = nn.ModuleList([nn.Linear(inp_dim*2, inp_dim, bias=False) for i in range(1, depth+1)])
        
        self.pre = nn.Sequential(
            Conv(3+2, inp_dim//8, 7, 2, bn=False, relu=True),
            Residual(inp_dim//8, inp_dim//4),
            Pool(2, 2),
            Residual(inp_dim//4, inp_dim//2),
            Residual(inp_dim//2, inp_dim)
        )
        
        self.hgs = Hourglass_history(depth, inp_dim, temp, bn, increase)
        self.features = nn.Sequential(Residual(inp_dim, inp_dim),
                                      Conv(inp_dim, inp_dim, 1, bn=False, relu=True)) 
        self.outs = Conv(inp_dim, oup_dim, 1, relu=False, bn=False)
        
    def forward(self, imgs, langs, lang_lengths, spatial_coords, time_idx, histories):
        batch_size = imgs.shape[0]
        imgs = torch.cat([imgs, spatial_coords], dim=1)
        
        ##########
        X = self.word_embedder(langs) # 16 50 D
        for l_i, length in enumerate(lang_lengths):
            X[:, length:, :] = 0.0
        
        X = torch.nn.utils.rnn.pack_padded_sequence(X, lang_lengths, batch_first=True)
        
        X, _ = self.word_rnn(X)
        
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True) # B T D
        
        word_embed = []
        for l_i, length in enumerate(lang_lengths):
            word_embed.append(X[l_i:l_i+1, :length, :]) # 1, B, T

        time_embed = self.time_embedder(time_idx).expand(batch_size, -1)
        
        x = self.pre(imgs)
        
        hg, curr_feat = self.hgs(x, word_embed,
                           histories, time_embed,
                           self.history_rnn,
                           self.W_w, self.W_h,
                           self.Ww_q, self.Ww_k, self.Ww_v,
                           self.Wh_q, self.Wh_k, self.Wh_v)

        feature = self.features(hg)
        pred = self.outs(feature)
        
        histories.append(torch.cat([curr_feat, time_embed], dim=1).view(batch_size, -1,
                                                                        curr_feat.shape[-1]+time_embed.shape[-1]))
                
        return pred, histories
    
        
class IT2P_nonhistory(nn.Module):
    def __init__(self, inp_dim, oup_dim, dictionary, word_embed_dim, temp, time_embed_dim=64, depth=4, t_length=8, bn=False, increase=0):
        super(IT2P_nonhistory, self).__init__()
        
        self.time_embedder = nn.Embedding(t_length, time_embed_dim)
                
        self.word_embed_dim = word_embed_dim
        self.word_embedder = nn.Embedding(len(dictionary), word_embed_dim)
        
        self.word_rnn = nn.LSTM(word_embed_dim, word_embed_dim, batch_first=True, num_layers=1, bidirectional=True)
        
        # set bias False
        self.Ww_q = nn.ModuleList([nn.Linear(inp_dim, inp_dim, bias=False) for i in range(1, depth+1)])
        self.Ww_k = nn.ModuleList([nn.Linear(word_embed_dim*2, inp_dim, bias=False) for i in range(1, depth+1)])
        self.Ww_v = nn.ModuleList([nn.Linear(word_embed_dim*2, inp_dim, bias=False) for i in range(1, depth+1)])
        
        self.pre = nn.Sequential(
            Conv(3+2, inp_dim//8, 7, 2, bn=False, relu=True),
            Residual(inp_dim//8, inp_dim//4),
            Pool(2, 2),
            Residual(inp_dim//4, inp_dim//2),
            Residual(inp_dim//2, inp_dim)
        )
        
        self.hgs = Hourglass_nonhistory(depth, inp_dim, temp, bn, increase)
        self.features = nn.Sequential(Residual(inp_dim, inp_dim),
                                      Conv(inp_dim, inp_dim, 1, bn=False, relu=True)) 
        self.outs = Conv(inp_dim, oup_dim, 1, relu=False, bn=False)
        
    def forward(self, imgs, langs, lang_lengths, spatial_coords):
        batch_size = imgs.shape[0]
        imgs = torch.cat([imgs, spatial_coords], dim=1)
        
        ##########
        X = self.word_embedder(langs) # 16 50 D
        for l_i, length in enumerate(lang_lengths):
            X[:, length:, :] = 0.0
        
        X = torch.nn.utils.rnn.pack_padded_sequence(X, lang_lengths, batch_first=True)
        
        X, _ = self.word_rnn(X)
        
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True) # B T D
        
        word_embed = []
        for l_i, length in enumerate(lang_lengths):
            word_embed.append(X[l_i:l_i+1, :length, :]) # 1, B, T

        x = self.pre(imgs)
        
        hg  = self.hgs(x, word_embed, self.Ww_q, self.Ww_k, self.Ww_v)

        feature = self.features(hg)
        pred = self.outs(feature)
                        
        return pred