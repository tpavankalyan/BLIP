import os
import json

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image
import pandas as pd

from data.utils import pre_caption
import torch

# from utils import pre_caption

class WIDO_train(Dataset):
    def __init__(self, train_path, transform, image_root, max_words=30, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        
        self.annotation = json.load(open(train_path,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        
        ann = self.annotation[index]
        #image_path = os.path.join(self.image_root,ann['image'])  
        image_path = self.image_root + ann['image']      
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']], image_path
        
    
class WIDO_retrieval_eval(Dataset):
    def __init__(self, val_path, test_path, transform, image_root, split, max_words=30):  
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''

        filenames = {'val':val_path,'test':test_path}

        self.annotation = json.load(open(filenames[split],'r'))
        self.transform = transform
        self.image_root = image_root
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            self.text.append(pre_caption(ann['caption'],max_words))
            self.img2txt[img_id].append(txt_id)
            self.txt2img[txt_id] = img_id
            txt_id += 1
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        image_path = self.image_root +  self.annotation[index]['image']        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  
        max_words = 30
        caption = pre_caption(self.annotation[index]['caption'],max_words)

        return image, caption, index, image_path
    
     
