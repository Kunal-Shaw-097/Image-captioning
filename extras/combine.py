import json
import os
from pathlib import Path

captions_dir = "data/Pascal_sentence"
save_json = os.path.join(captions_dir, "captions.json")


if Path(save_json).exists() :
    os.remove(save_json)
    
data = []
for file in os.listdir(captions_dir) :
    if file.endswith(".json") :
        with open(os.path.join(captions_dir, file), "r") as f :
            temp = json.load(f)
            data += temp

with open(save_json, "w") as f :
    json.dump(data, f, indent= True)