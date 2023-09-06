from torchsummary import summary
from utils.modules import decoder_block, encoder_backbone_up,encoder_backbone_down, encoder_head
from utils.model import Decoder, Encoder, Model
import torch.nn as nn
import torch

input = torch.rand((1, 3, 640, 640)).cuda()


model = Encoder(3, 512, depth_ratio= 0.33).cuda()
#model = Decoder(1000, 512, 8, 6).cuda()
#model = Model(1000, 3, 512, 8 , 6).cuda()

print(summary(model,(3, 640, 480), device= "cuda"))
#print(summary(model,[(1,64, 512), (1,64, 512)], device= "cuda"))
#print(summary(model,[(1,64, 512), (3, 640,640)], device= "cuda"))

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")
