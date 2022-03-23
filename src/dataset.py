from torch.utils.data import DataLoader, Dataset
import os
import json
import random
import numpy as np
import torch
import torchvision
from PIL import Image

class HDMdataset(Dataset):
    def __init__(self, data_dir, split, split_info, dictionary, is_train, is_seq=True, max_len=50):
        self.data_dir = data_dir
        self.split = split
        self.split_info = split_info
        self.dictionary = dictionary
        self.is_seq = is_seq
        self.max_len = max_len
        
        if split != -1:
            self.curr_test_idx = split_info[str(split)][0]
            self.curr_train_idx = [i for i in range(300) if i not in self.curr_test_idx]
        elif split == -1:
            print('use all dataset for training')
            self.curr_test_idx = []
            self.curr_train_idx = [i for i in range(300)]
        
        self.curr_scene_fps = list()
        if is_train:
            self.curr_target_idx = self.curr_train_idx
        else:
            self.curr_target_idx = self.curr_test_idx
            
        for idx in self.curr_target_idx:
            self.curr_scene_fps.append(os.path.join(data_dir, 'scene_{:04}'.format(idx)))    
            
        self.transform = torchvision.transforms.ToTensor()
        self.resizer = torchvision.transforms.Resize(64)
            
    def __len__(self):
        return len(self.curr_scene_fps)
    
    def get_sample(self, curr_scene_fp, curr_meta_fp, HFLIP):
        start_idx, end_idx = curr_meta_fp[:3].split('_')
        start_img_fp = os.path.join(curr_scene_fp, 'image', 'bird{:06}.png'.format(int(start_idx)))
        front_img_fp = os.path.join(curr_scene_fp, 'image', 'front{:06}.png'.format(int(start_idx)))
        end_img_fp = os.path.join(curr_scene_fp, 'image', 'bird{:06}.png'.format(int(end_idx)))
        pick_map_fp = os.path.join(curr_scene_fp, 'heatmap', 'pick_{}_{}.jpg'.format(start_idx, end_idx))
        place_map_fp = os.path.join(curr_scene_fp, 'heatmap', 'place_{}_{}.jpg'.format(start_idx, end_idx))

        start_img = self.transform(Image.open(start_img_fp))[:3]
        front_img = self.transform(Image.open(front_img_fp))[:3]
        end_img = self.transform(Image.open(end_img_fp))[:3]
        pick_map = self.transform(self.resizer(Image.open(pick_map_fp)))[0]
        place_map = self.transform(self.resizer(Image.open(place_map_fp)))[0]

        if HFLIP:
            start_img = torchvision.transforms.functional.hflip(start_img)
            front_img = torchvision.transforms.functional.hflip(front_img)
            end_img = torchvision.transforms.functional.hflip(end_img)
            pick_map = torchvision.transforms.functional.hflip(pick_map)
            place_map = torchvision.transforms.functional.hflip(place_map)

        meta = json.load(open(os.path.join(curr_scene_fp, 'meta', curr_meta_fp), 'r'))

        curr_sentence = random.choice(meta['sentence'])
        curr_explicit = meta['explicit'][meta['sentence'].index(curr_sentence)]
        curr_implicit = meta['implicit'][meta['sentence'].index(curr_sentence)]
        
        if HFLIP:
            orig_sentence = curr_sentence
            curr_sentence = curr_sentence.replace('right', 'qwerty')
            curr_sentence = curr_sentence.replace('left', 'asdfgh')
            curr_sentence = curr_sentence.replace('qwerty', 'left')
            curr_sentence = curr_sentence.replace('asdfgh', 'right')
            meta['sentence'][meta['sentence'].index(orig_sentence)] = curr_sentence

        curr_word_index = torch.LongTensor([0 for _ in range(self.max_len)])
        sentence_length = len(curr_sentence.split())
        curr_word_index[:sentence_length] = torch.LongTensor([self.dictionary[w] for w in curr_sentence.split()])

        sample = dict()
        sample['start_img'] = start_img * 2 - 1
        sample['end_img'] = end_img * 2 - 1
        sample['pick_map'] = pick_map * 2 -1
        sample['place_map'] = place_map * 2 -1
        sample['front_img'] = front_img
        sample['sentence'] = curr_word_index
        sample['sentence_length'] = sentence_length
        sample['orig_sentence'] = curr_sentence
        sample['explicit'] = curr_explicit
        sample['implicit'] = curr_implicit
            
        return sample
    
    
    def __getitem__(self, idx):
        curr_scene_fp = self.curr_scene_fps[idx]
        samples = list()
        
        HFLIP = False
        if np.random.rand() > 0.5:
            HFLIP = True
        
        if self.is_seq:
            for curr_meta_fp in sorted(os.listdir(os.path.join(curr_scene_fp, 'meta'))):
                samples.append(self.get_sample(curr_scene_fp, curr_meta_fp, HFLIP))
            return samples
        
        else:
            curr_meta_fp = random.choice(os.listdir(os.path.join(curr_scene_fp, 'meta')))
            return self.get_sample(curr_scene_fp, curr_meta_fp, HFLIP)
                                   
                                   
