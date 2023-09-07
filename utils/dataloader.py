import torch
import numpy as np
from pathlib import Path
import shutil
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import cv2
import os
import json
from torch.utils.data import Dataset
from utils.process_image import process_img_batch
from utils.tokenizer import Tokenizer
import time




class COCO_Dataset(Dataset) :
    def __init__(self, img_root_path, caption_path, vocab_path, mask : bool = False) :  #number of workers is for caching images
        super().__init__() 

        self.ims , self.y = self.load_data(img_root_path, caption_path)
        self.n = len(self.ims)
        self.tokenizer = Tokenizer(vocab_path) 
        self.mask = mask
        
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
        x = process_img_batch(list(x))
        y = self.tokenizer.encode(list(y), mask =self.mask)
        return x, y[0], y[1]    # return img, encoded sentences, mask


                                                      
    

    


