import torch.nn as nn
import torch

def train_history(model, optimizer, samples, spatial_coords, device='cuda'):
    histories = []
    
    mse = nn.MSELoss(reduction='mean')

    loss = 0.0
    for i, sample in enumerate(samples):
        sort_idx = torch.argsort(sample['sentence_length'], descending=True)
        sample['sentence'] = sample['sentence'][sort_idx]
        sample['start_img'] = sample['start_img'][sort_idx]
        sample['pick_map'] = sample['pick_map'][sort_idx]
        sample['place_map'] = sample['place_map'][sort_idx]
        sample['sentence_length'] = sample['sentence_length'][sort_idx]

        imgs = sample['start_img'].float().to(device)
        langs = sample['sentence'].long().to(device)
        lang_lengths = sample['sentence_length'].long()
        time = torch.LongTensor([i]).to(device)
        
        pred, histories = model(imgs, langs, lang_lengths, spatial_coords, time, histories)
        
        pick_gt = sample['pick_map'].to(device)
        place_gt = sample['place_map'].to(device)
        
        loss += mse(pred[:, 0, :, :], pick_gt)
        loss += mse(pred[:, 1, :, :], place_gt)
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    del histories
    histories = []
    torch.cuda.empty_cache()
    
    return loss.item()

def train_nonhistory(model, optimizer, samples, spatial_coords, device='cuda'):    
    mse = nn.MSELoss(reduction='mean')

    loss = 0.0
    for i, sample in enumerate(samples):
        sort_idx = torch.argsort(sample['sentence_length'], descending=True)
        sample['sentence'] = sample['sentence'][sort_idx]
        sample['start_img'] = sample['start_img'][sort_idx]
        sample['pick_map'] = sample['pick_map'][sort_idx]
        sample['place_map'] = sample['place_map'][sort_idx]
        sample['sentence_length'] = sample['sentence_length'][sort_idx]

        imgs = sample['start_img'].float().to(device)
        langs = sample['sentence'].long().to(device)
        lang_lengths = sample['sentence_length'].long()
        
        pred = model(imgs, langs, lang_lengths, spatial_coords)
        
        pick_gt = sample['pick_map'].to(device)
        place_gt = sample['place_map'].to(device)
        
        loss += mse(pred[:, 0, :, :], pick_gt)
        loss += mse(pred[:, 1, :, :], place_gt)
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
