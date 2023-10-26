import cv2
import json
from pathlib import Path
import shutil
import os
from uuid import uuid4

image_dir = "test_images"

image_save_dir = Path("data/Custom_Dataset/images")

num_captions = 3


if __name__ == "__main__" :

    if image_save_dir.exists():
        shutil.rmtree(image_save_dir)

    image_save_dir.mkdir(parents=True)

    captions_list = []

    images = os.listdir(image_dir)
    num_images = len(images)

    for i, img_name in enumerate(images) :
        img = cv2.imread(os.path.join(image_dir, img_name))
        save_name = f"{uuid4()}.jpg"
        cv2.imshow(" ", img)

        if cv2.waitKey(0) & 0xFF == ord('s'):
            for j in range(num_captions) :
                caption = input(f"(Image {i + 1} of total {num_images}) Enter caption: {j}: ")
                obj = {
                    "image_id" : save_name,
                    "caption" : caption
                }
                captions_list.append(obj)
            cv2.imwrite(os.path.join(image_save_dir, save_name), img)
            with open(image_save_dir.parent / "captions.json", "w") as f :
                json.dump(captions_list, f, indent= True)
            print(" ")

        elif cv2.waitKey(0) & 0xFF == ord('q'):
            continue 
        elif cv2.waitKey(0) & 0xFF == ord('e'):
            exit() 
     






