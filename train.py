import torch
from torch.utils.data import DataLoader
from utils.dataloader import COCO_Dataset
from utils.tokenizer import Tokenizer
from utils.model import Model
from tqdm import tqdm
from copy import deepcopy

TQDM_BAR_FORMAT = "{l_bar}{bar:15}{r_bar}"
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul 
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

batch_size = 16
num_epochs = 10
learning_rate = 0.0001

train_img_root_path = "/mnt/DA149E26149E0623/data/COCO_dataset/train2017"
val_img_root_path = "/mnt/DA149E26149E0623/data/COCO_dataset/val2017"

train_caption_path = "/mnt/DA149E26149E0623/data/COCO_dataset/captions_train2017.json"
val_caption_path = "/mnt/DA149E26149E0623/data/COCO_dataset/captions_val2017.json"

vocab_path = "vocab.json"


train_dataset = COCO_Dataset(train_img_root_path, train_caption_path, vocab_path)
val_dataset = COCO_Dataset(val_img_root_path, val_caption_path, vocab_path)

train_dataloader = DataLoader(
    train_dataset, pin_memory= True, batch_size= batch_size,
    shuffle= True, num_workers= 8, collate_fn= train_dataset.custom_collate_fn)

val_dataloader = DataLoader(
    val_dataset, pin_memory= True,batch_size= batch_size,
    shuffle = True, num_workers= 8, collate_fn= val_dataset.custom_collate_fn)

tokenizer = Tokenizer(vocab_path)
model = Model(len(tokenizer), 3, 512, 8, 6, 0.1).cuda()


if __name__ == "__main__" : 
    
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor= 1.0, end_factor= 0.05, total_iters= num_epochs//2)
    scaler = torch.cuda.amp.GradScaler(enabled= True)
    
    train_steps = len(train_dataloader)
    val_steps = len(val_dataloader)
    
    
    train_losses_per_epoch = []
    val_losses_per_epoch = []
    
    for _ in range(num_epochs): 
        
        best_average_loss = float('inf')
        train_totol_loss= 0
        val_total_loss = 0
        
        model.train()
        train_pbar = tqdm(enumerate(train_dataloader), total = train_steps, bar_format= TQDM_BAR_FORMAT)
        
        for i, batch in train_pbar: 
                        
            x , y, target = batch
            x = x.cuda(non_blocking=True).float()/255
            y = y.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        
            with torch.cuda.amp.autocast(): 
                output = model(x, y)
            
            loss = model.loss(output, target)
            
            if i != 0 and (i % 500 == 0) or i == train_steps -1 :
                print(tokenizer.decode(model.process_output(output)))

            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad()
            
            loss= loss.item()
            train_totol_loss += loss
            avg_train_loss = train_totol_loss/(i + 1)
            train_pbar.set_description(f"TRAINING Epoch {_} : Average loss : {avg_train_loss:03f}, loss : {loss:03f}")

        model.eval()
        val_pbar = tqdm(enumerate(val_dataloader), total = val_steps, bar_format= TQDM_BAR_FORMAT)
        
        with torch.no_grad() :
            
            for i, batch in val_pbar:
                
                x , y, target = batch
                x = x.cuda(non_blocking=True).float()/255
                y = y.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            
                output = model(x, y)
                    
                loss = model.loss(output, target)
                
                loss= loss.item()
                val_total_loss += loss
                avg_val_loss = val_total_loss/(i + 1)
                val_pbar.set_description(f"VALIDATION Epoch {_} : Average loss : {avg_val_loss:03f}, loss : {loss:03f}")
        
            if avg_val_loss < best_average_loss :
                best_avg_loss = avg_val_loss
                ckpt = {
                'epoch': _,
                'model': deepcopy(model),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_losses' : train_losses_per_epoch,
                'val_losses' : val_losses_per_epoch}
        
        train_losses_per_epoch.append(avg_train_loss)
        val_losses_per_epoch.append(avg_val_loss)
        
