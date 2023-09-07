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


train_dataset = COCO_Dataset(train_img_root_path, train_caption_path, vocab_path, mask= False)
#val_dataset = COCO_Dataset(val_caption_path)

train_dataloader = DataLoader(
    train_dataset, pin_memory= True, batch_size= batch_size,
    shuffle= True, num_workers= 4, collate_fn= train_dataset.custom_collate_fn)

#val_dataloader = DataLoader(val_dataset, batch_size= batch_size, shuffle = True)

tokenizer = Tokenizer(vocab_path)
model = Model(len(tokenizer), 3, 512, 8, 6).cuda()

#model = torch.compile(model, dynamic= True)
def check(x) :
    return x.is_contiguous()

if __name__ == "__main__" : 

    optimizer = torch.optim.Adam(model.parameters(), 0.0001)
    scaler = torch.cuda.amp.GradScaler(enabled= True)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


    prof = torch.profiler.profile(
        activities= [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],   
        schedule=torch.profiler.schedule(wait=10, warmup=10, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/test'),
        profile_memory=True, record_shapes=True, with_stack=True, with_modules= True,
        ) 

    prof.start()
    losses = []
    avg_loss = []
    n = len(train_dataloader)
    for i,batch in tqdm(enumerate(train_dataloader), total = n) : 
    #for i,batch in enumerate(train_dataloader) :

        x, y, mask = batch
        x = x.cuda(non_blocking=True).float()/255
        y = y.cuda(non_blocking=True)
        y_in , y_out = y[: , :-1].contiguous(), y[:,1:].contiguous()
        print(check(y_in))
        break
        '''if mask :
            mask = mask.cuda(non_blocking=True)

        with torch.cuda.amp.autocast(): 
            output = model(x, y, mask)
        
        loss = model.loss(output, y)
        avg_loss.append(loss.item())
        if (i % 500 == 0) or i == n-1:
            print(tokenizer.decode(model.process_output(output)))
            losses.append(sum(avg_loss)/len(avg_loss))
            avg_loss = []
        optimizer.zero_grad()

        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

        scaler.step(optimizer)
        scaler.update()'''

        #prof.step()
    print(losses)
    prof.stop()