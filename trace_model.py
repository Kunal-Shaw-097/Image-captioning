from utils.general import resume_checkpoint
from utils.tokenizer import Tokenizer
import torch


model_path = "saved_model/best.pt"
vocab_path = "vocab.json"

tokenizer = Tokenizer(vocab_path)

model = resume_checkpoint(model_path, tokenizer= tokenizer, device="cpu")

model.eval()

traced_model = torch.jit.trace(model, (torch.ones(1, 3, 480, 480), torch.ones(1,10).type(torch.int64)))

torch.jit.save(traced_model, "saved_model/traced_best.pt")

