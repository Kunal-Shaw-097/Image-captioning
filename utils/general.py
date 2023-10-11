import torch
from utils.model import Model

def resume_checkpoint(path, tokenizer, resume_weight_only : bool = True) :
    ckpt = torch.load(path)
    model = Model(len(tokenizer), **ckpt["model_args"]).cuda()
    model.load_state_dict(ckpt["model"])
    if resume_weight_only :
        return model
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler = Custom_LinearScheduler(**ckpt["scheduler"])
    start_epoch = ckpt["epoch"] + 1
    num_epochs= scheduler.total_epochs
    train_losses, val_losses = ckpt["train_losses"], ckpt['val_losses']
    return model, ckpt["model_args"], optimizer, scheduler, start_epoch, num_epochs, train_losses, val_losses

def strip_weight(path):
    ckpt = torch.load(path)
    new_ckpt = {
        "model" : ckpt["model"],
        "model_args" : ckpt["model_args"]
    }
    torch.save(new_ckpt, path)
    print(f"Stipped weight from : {path}")
    return


class Custom_LinearScheduler():
    def __init__(self, initial_lr, min_lr, warmup_steps, total_epochs):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.warmup_steps = warmup_steps

    def get_warmup_lr(self, step) :
        return self.initial_lr * (step/self.warmup_steps)
    
    def get_lr(self, epoch):
        factor = (self.total_epochs - epoch + 1)/self.total_epochs
        return max(self.min_lr, self.initial_lr * factor)
    
    def state_dict(self):
        ckpt = {
            "initial_lr" : self.initial_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "total_epochs": self.total_epochs,
        }
        return ckpt