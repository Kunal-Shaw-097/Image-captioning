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
    def __init__(self, img_root_path, caption_path, vocab_path, cache_disk : bool = True, num_workers : int = 4) :  #number of workers is for caching images
        super().__init__()
        assert isinstance(num_workers, int), "number of workers must be an integer"
        if os.cpu_count() - 1 < num_workers :
            num_workers = os.cpu_count() - 1
            print("Capping num_workers to max cpu counts")

        self.cache_save_path = Path(img_root_path).parent / 'cached_images'  
        
        self.ims , self.y = self.load_data(img_root_path, caption_path)
        
        self.n = len(self.ims)
        
        self.cache = cache_disk
        if self.cache :
            time1 = time.perf_counter()
            self.cache_ims = [str(self.cache_save_path / Path(ims).stem ) + ".npy" for ims in self.ims]
            print(time.perf_counter() - time1)
            if not self.cache_save_path.is_dir() :
                iterator = ThreadPool(num_workers).imap(self.load_img, set(self.ims))
                self.cache_save_path.mkdir()
                print("Caching images to " + f"{ self.cache_save_path }")
                for x in  tqdm(iterator, total = len(set(self.ims))) : 
                    im, name = x
                    name = name + '.npy'
                    np.save(self.cache_save_path / name, im)
                print("Images Cached to " + str(self.cache_save_path.as_posix()))


        self.tokenizer = Tokenizer(vocab_path)

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

    def load_img(self, filename):
        return cv2.imread(filename) , Path(filename).stem
    
    def __len__(self) :
        return self.n
    
    def __getitem__(self, index):
        if self.cache :
            return  self.cache_ims[index], self.y[index]
        return self.ims[index], self.y[index]
    
    def collate_fn(self, batch):
        x, y = zip(*batch)
        if self.cache :
            x = process_img_batch(list(x), cached = True)
        else :
            x = process_img_batch(list(x), cached = False)
        y = self.tokenizer.encode(list(y))
        return x, y[0], y[1]                                                            # return img, encoded sentences, mask
    

    


