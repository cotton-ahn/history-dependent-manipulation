import torch
from skimage.transform import resize

def test_history(model, samples, spatial_coords, device='cuda'):
    model.eval()
    total_preds = {'pick':[], 'place':[]}
    histories = []
    
    for time, sample in enumerate(samples):
        imgs = sample['start_img'].float().to(device)
        langs = sample['sentence'].long().to(device)
        lang_lengths = sample['sentence_length'].long()
        time = torch.LongTensor([time]).to(device)
        
        with torch.no_grad():
            pred, histories = model(imgs, langs, lang_lengths, spatial_coords, time, histories)
            
        total_preds['pick'].append(resize(pred[:, 0, :, :].squeeze().detach().cpu().numpy(), (256, 256)))
        total_preds['place'].append(resize(pred[:, 1, :, :].squeeze().detach().cpu().numpy(), (256, 256)))
        
    del histories
    histories = []
    torch.cuda.empty_cache()
    
    return total_preds

def test_nonhistory(model, sample, spatial_coords, device='cuda'):
    model.eval()
    total_preds = {'pick':[], 'place':[]}
    
    imgs = sample['start_img'].float().to(device)
    langs = sample['sentence'].long().to(device)
    lang_lengths = sample['sentence_length'].long()
    with torch.no_grad():
        pred = model(imgs, langs, lang_lengths, spatial_coords)
    total_preds['pick'] = (resize(pred[:, 0, :, :].squeeze().detach().cpu().numpy(), (256, 256)))
    total_preds['place'] = (resize(pred[:, 1, :, :].squeeze().detach().cpu().numpy(), (256, 256)))
        
    return total_preds
