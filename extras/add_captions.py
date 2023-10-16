import cv2
import json
from pathlib import Path
import shutil
import os
from uuid import uuid4

image_dir = "test_images"

image_save_dir = Path("data/Custom_Dataset/images")

num_captions = 5


if __name__ == "__main__" :

    if image_save_dir.exists():
        shutil.rmtree(image_save_dir)

    image_save_dir.mkdir(parents=True)

    captions_list = []

    for i, img_name in enumerate(os.listdir(image_dir)) :
        img = cv2.imread(os.path.join(image_dir, img_name))
        save_name = uuid4() + ".jpg"
        cv2.imwrite(os.path.join(image_dir, save_name), img)
        cv2.imshow(" ", img)
        if cv2.waitKey(0) & 0xFF == ord('s'):
            for i in range(num_captions) :
                caption = input(f"Enter caption: {i}")
                obj = {
                    "image_id" : save_name,
                    "caption" : caption
                }
                captions_list.append(obj)
    






