import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.dataloader import COCO_Dataset
from utils.tokenizer import Tokenizer
from utils.model import Model
from tqdm import tqdm
import time




batch_size = 16

train_img_root_path = "data/COCO_dataset/train2017"
val_img_root_path = "data/COCO_dataset/val2017"

train_caption_path = "data/COCO_dataset/captions_train2017.json"
val_caption_path = "data/COCO_dataset/captions_val2017.json"

vocab_path = "vocab.json"


train_dataset = COCO_Dataset(train_img_root_path, train_caption_path, vocab_path, cache_disk= False)
#val_dataset = COCO_Dataset(val_caption_path)

train_dataloader = DataLoader(train_dataset, pin_memory= True, batch_size= batch_size, shuffle= True, num_workers= 4, collate_fn= train_dataset.collate_fn)
#val_dataloader = DataLoader(val_dataset, batch_size= batch_size, shuffle = True)

tokenizer = Tokenizer(vocab_path)
model = Model(len(tokenizer), 3, 512, 8, 6).cuda()

if __name__ == "__main__" : 

    optimizer = torch.optim.Adam(model.parameters(), 0.0001)
    scaler = torch.cuda.amp.GradScaler(enabled= True)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=10, warmup=10, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/test'),
        profile_memory=True, record_shapes=True, with_stack=True, with_modules= True,
        ) as prof:

        #for i,batch in tqdm(enumerate(train_dataloader), total = len(train_dataloader)) : 
        for i,batch in enumerate(train_dataloader) :
            if i > 30:
                break

            x, y, mask = batch
            x = x.cuda(non_blocking=True).float()/255
            y = y.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            with torch.cuda.amp.autocast():
                output = model(x, y, mask)
            
            loss = model.loss(output, y)

            optimizer.zero_grad()

            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

            scaler.step(optimizer)
            scaler.update()

            prof.step()