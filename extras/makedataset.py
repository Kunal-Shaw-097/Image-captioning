import os 
import cv2
import json
from pathlib import Path
import shutil



save_dir_imgs = Path("data/images")

if save_dir_imgs.exists() :
    shutil.rmtree(save_dir_imgs.parent)

save_dir_imgs.mkdir(parents= True)

images = "dataset"
labels = "sentence"


for folder in os.listdir(images) :
    for image in os.listdir(os.path.join(images,folder)):
        img = cv2.imread(os.path.join(images,folder,image))
        image_name = "".join(image.split(".")[:-1]) + ".jpg"
        cv2.imwrite(os.path.join(save_dir_imgs, image_name), img)

captions = []

for folder in os.listdir(labels) :
    for file in os.listdir(os.path.join(labels, folder)):
        with open(os.path.join(labels, folder, file), "r") as f :
            temp = f.readlines()    
        for caption in temp :
            obj = {
                "image_id" : file[:-3] + ".jpg",
                "caption" : caption[:-1]
            }
            captions.append(obj)

with open(os.path.join(save_dir_imgs.parent, "captions.json"), "w") as f:
    json.dump(captions, f, indent= True)