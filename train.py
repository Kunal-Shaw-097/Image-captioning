import torch
from torch.utils.data import DataLoader, ConcatDataset
from utils.dataloader import COCO_Dataset, Flicker30_dataset, Custom_Captions_dataset, Chunks
from utils.tokenizer import Tokenizer
from utils.model import Model
from utils.general import Custom_LinearScheduler, resume_checkpoint, strip_weight
from tqdm import tqdm
from pathlib import Path
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

TQDM_BAR_FORMAT = "{l_bar}{bar:15}{r_bar}"
torch.use_deterministic_algorithms = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul 
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

COCO_train_img_root_path = "data/COCO_dataset/train2017"
Flicker_train_img_root_path = "data/Flicker30_dataset/flickr30k_images"
ConceptaulCaptions_train_img_root_path = "data/conceptual_captions/images"
PascalSentence_train_img_root_path = "data/Pascal_sentence/images"
COCO_val_img_root_path = "data/COCO_dataset/val2017"
lian_COCO_img_path = "data/lian_coco/images"


COCO_train_caption_path = "data/COCO_dataset/captions_train2017.json"
Flicker_train_caption_path = "data/Flicker30_dataset/captions.csv"
ConceptaulCaptions_caption_path = "data/conceptual_captions/captions.json"
PascalSentence_caption_path = "data/Pascal_sentence/captions.json"
COCO_val_caption_path = "data/COCO_dataset/captions_val2017.json"
lian_COCO_caption_path = "data/lian_coco/final_captions.json"


vocab_path = "vocab.json"

model_save_dir =Path(__file__).parent / "saved_model"

resume_checkpoint_path = "saved_model/epoch1.pt"
resume_training = True
resume_weight_only = True

augment_data = True
save_per_epoch = True
img_size = 480
num_workers = 4
batch_size = 32
gradient_accumulation_steps = 8
num_epochs = 3

initial_lr = 5e-5   # starting learning rate
min_lr = 2e-5       # lowest learning rate
warmup_steps = 10000

model_args = {

    "in_channels": 3,
    "patch_size": 30,
    "emb_dim": 768,
    "num_heads": 12,
    "num_layers": 6,
    "dropout": 0.25,
}


if __name__ == "__main__" :
    
    set_seed(420)

    COCO_train_dataset = COCO_Dataset(COCO_train_img_root_path, COCO_train_caption_path, vocab_path, img_size, augment_data)
    Flicker_train_dataset = Flicker30_dataset(Flicker_train_img_root_path, Flicker_train_caption_path, vocab_path, img_size, augment_data)
    PascalSentence_dataset = Custom_Captions_dataset(PascalSentence_train_img_root_path, PascalSentence_caption_path, vocab_path, img_size, augment_data)
    ConceptualCaptions_train_dataset = Custom_Captions_dataset(ConceptaulCaptions_train_img_root_path , ConceptaulCaptions_caption_path , vocab_path, img_size, augment_data)
    lian_COCO_train_dataset = Custom_Captions_dataset(lian_COCO_img_path, lian_COCO_caption_path,vocab_path, img_size, augment_data)
    
   
    val_dataset = COCO_Dataset(COCO_val_img_root_path, COCO_val_caption_path, vocab_path)
    train_dataset = ConcatDataset([COCO_train_dataset, Flicker_train_dataset, PascalSentence_dataset, ConceptualCaptions_train_dataset, lian_COCO_train_dataset]) 

    val_dataloader = DataLoader(
        val_dataset, pin_memory= True,batch_size= batch_size,
        shuffle = True, num_workers= num_workers, collate_fn= val_dataset.custom_collate_fn)
    
    tokenizer = Tokenizer(vocab_path)
    
    if resume_training and Path(resume_checkpoint_path).exists and not resume_weight_only:
        model, model_args, optimizer, scheduler, start_epoch, num_epochs, train_losses, val_losses = resume_checkpoint(resume_checkpoint_path, tokenizer, resume_weight_only)
        best_avg_loss = min(val_losses)
        print(f"Resuming training from epoch: {start_epoch}")
    else :
        if resume_training == True and Path(resume_checkpoint_path).exists == False :
            print("No valid Path for checkpoint found ! Starting the training from beginning.")
            model = Model(vocab_size= len(tokenizer), **model_args).cuda()
        elif resume_training == True and resume_weight_only :
            print("Starting new training from pretrained checkpoint.")
            model = resume_checkpoint(resume_checkpoint_path, tokenizer, resume_weight_only)
            warmup_steps = 0
        else :  
            model = Model(vocab_size= len(tokenizer), **model_args).cuda()
        optimizer = torch.optim.Adam(model.parameters(), initial_lr, fused= True)
        scheduler = Custom_LinearScheduler(initial_lr, min_lr, warmup_steps, num_epochs)
        start_epoch= 1
        train_losses = []
        val_losses = []
        best_avg_loss = float('inf')
    
    scaler = torch.cuda.amp.GradScaler(enabled= True)
    
    train_steps = len(train_dataset)//batch_size
    val_steps = len(val_dataloader) 

    for _ in range(start_epoch, num_epochs + 1): 

        i = 0
        
        train_totol_loss= 0
        val_total_loss = 0
        
        model.train()
        train_pbar = tqdm(total = train_steps, bar_format= TQDM_BAR_FORMAT, colour= "#00ff00")

        lr = scheduler.get_lr(_)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train_dataset_chunks = Chunks(train_dataset)
        train_dataset_iterable = iter(train_dataset_chunks)

        for c in range(len(train_dataset_chunks.lengths)) :

            train_dataloader = DataLoader(
                next(train_dataset_iterable), pin_memory= True, batch_size= batch_size,
                shuffle= True, num_workers= num_workers, collate_fn= COCO_train_dataset.custom_collate_fn)
        
            for batch in train_dataloader:

                if i < warmup_steps and _ == 1 :
                    lr = scheduler.get_warmup_lr(i + 1)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                x , y, target = batch
                x = x.cuda(non_blocking=True).float()/255
                y = y.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            
                with torch.cuda.amp.autocast(dtype= torch.bfloat16): 
                    output, loss = model(x, y, target)
                    loss = loss/gradient_accumulation_steps

                scaler.scale(loss).backward()

                '''if i != 0 and i % 10000 == 0 :
                    print(tokenizer.decode(target))
                    print(tokenizer.decode(model.process_output(output)))'''
                
                if (i + 1) % gradient_accumulation_steps == 0 or i == train_steps -1 :

                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    scaler.step(optimizer)
                    scaler.update()
                    
                    optimizer.zero_grad()
                
                i += 1
                train_pbar.update(1)
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

                with torch.cuda.amp.autocast(dtype= torch.bfloat16):
                    output, loss = model(x, y, target)
                
                loss= loss.item()
                if loss > 0:
                    val_total_loss += loss
                    avg_val_loss = val_total_loss/(i + 1)

                val_pbar.set_description(f"VALIDATION Epoch {_} : Average loss : {avg_val_loss:03f}, loss : {loss:03f}")
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(" ")
        
        if avg_val_loss < best_avg_loss and avg_val_loss != 0:
            best_avg_loss = avg_val_loss
            ckpt = {
                'epoch': _,
                'model_args': model_args,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 
                'scheduler': scheduler.state_dict(),
                'train_losses' : train_losses,
                'val_losses' : val_losses,
                }
            if not model_save_dir.exists():
                model_save_dir.mkdir(parents= True, exist_ok= True)
            torch.save(ckpt, model_save_dir / "best.pt")
            del ckpt
            
        
        if save_per_epoch :
            ckpt = {
                'epoch': _,
                'model_args': model_args,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_losses' : train_losses,
                'val_losses' : val_losses,
                }
            
            if not model_save_dir.exists():
                model_save_dir.mkdir(parents= True, exist_ok= True)
            torch.save(ckpt, model_save_dir / f"epoch{_}.pt")
            del ckpt
    
    torch.cuda.empty_cache()
    ckpt = torch.load(model_save_dir / "best.pt")
    best_epoch = ckpt["epoch"]
    print(f"Best model is epoch: {best_epoch}")
    del ckpt

    strip_weight(model_save_dir / "best.pt")
    strip_weight(model_save_dir/ f"epoch{num_epochs}.pt")

        