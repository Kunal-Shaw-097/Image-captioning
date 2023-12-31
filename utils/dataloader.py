import os
import json
from typing import Iterator
from torch.utils.data import Dataset, IterableDataset
from utils.process_image import process_img_batch
from utils.tokenizer import Tokenizer
import albumentations as A
import torch
import numpy as np
import pandas as pd


class COCO_Dataset(Dataset) :
    def __init__(self, img_root_path, caption_path, vocab_path, img_size : int = 512, transform : bool = False) : 
        super().__init__() 

        self.ims , self.y = self.load_data(img_root_path, caption_path)
        self.n = len(self.ims)
        self.tokenizer = Tokenizer(vocab_path)
        self.transform = transform
        self.img_size = img_size
        if self.transform :
            self.augment = A.Compose([
                A.RandomBrightnessContrast(0.3, 0.3, p = 0.3),
                A.Blur(3, p = 0.3),
                A.Rotate(20, p = 0.3)
            ], p = 1)
            
    def load_data(self, image_root_path, path) :
        with open(path, 'r') as stream :
            data = json.load(stream)
            data = data["annotations"]
        x = []
        y = []
        for obj in data :
            image_id = obj["image_id"]
            x.append(os.path.join(image_root_path, f"{image_id:012d}.jpg"))
            y.append(obj["caption"])
        return x, y
    
    def __len__(self) :
        return self.n
    
    def __getitem__(self, index):
        return self.ims[index], self.y[index]
    
    def custom_collate_fn(self, batch):
        x, y = zip(*batch)
        x = process_img_batch(list(x), self.img_size)
        if self.transform :
            im = []
            for img in x :
                img = self.augment(image =img)["image"]
                im.append(img)
            x = torch.from_numpy(np.array(im).transpose(0, 3, 1, 2)).contiguous()
        else :
            x = torch.from_numpy(np.array(x).transpose(0, 3, 1, 2)).contiguous() 
        y = self.tokenizer.encode(list(y))
        input_y , target_y = y[: , :-1].contiguous(), y[:,1:].contiguous()
        return x, input_y, target_y         # return img, input encoded sentences, target encoded sentences
    
    
class Custom_Captions_dataset(COCO_Dataset):
    def __init__(self, img_root_path, caption_path, vocab_path, img_size : int = 512, transform: bool = False):
        super().__init__(img_root_path, caption_path, vocab_path, img_size, transform)
    
    def load_data(self, image_root_path, path):
        with open(path, 'r') as stream :
            data = json.load(stream)
        x = []
        y = []
        for obj in data :
            image_id = obj["image_id"]
            x.append(os.path.join(image_root_path, image_id))
            y.append(obj["caption"])
        return x, y
  

class Flicker30_dataset(COCO_Dataset):

    def __init__(self, img_root_path, caption_path, vocab_path, img_size : int = 512, transform: bool = False):
        super().__init__(img_root_path, caption_path, vocab_path, img_size, transform)
        
    def load_data(self, image_dir, caption_path):
        data = pd.read_csv(caption_path, delimiter= "|")
        x = [os.path.join(image_dir, i ) for i in data["image_name"].to_list()]
        y = data["comment"].to_list()
        return x, y 
    

class Chunks(IterableDataset) :

    def __init__(self, data, lengths : list= [0.25, 0.25, 0.25, 0.25]) :
        super().__init__()
        self.data = data
        self.lengths = lengths

    def __iter__(self) :
        chunks = torch.utils.data.random_split(self.data, self.lengths)
        for i in range(len(self.lengths)) :
            yield chunks[i]



            
    




    





                                                      
    

    


