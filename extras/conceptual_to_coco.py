"""Python scirpt to convert Coneptual caption dataset to coco format."""
import pandas as pd
import io
from PIL import Image
from pathlib import Path
import shutil
from uuid import uuid4
import json
import math
import asyncio
import aiohttp
from tqdm import tqdm


parquet_file_path = "Parquets/0000.parquet"
image_save_dir = Path("data/lian_coco_1/images")
image_per_batch = 10000


headers = {
    #'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
    'User-Agent':'Googlebot-Image/1.0', # Pretend to be googlebot
    'X-Forwarded-For': '64.18.15.200'
}


async def download_image(session, semaphore, x, retries = 0) :
    url , caption = x[0], x[1]
    try :
        async with semaphore :
            async with session.get(url, timeout = 2) as response:
                if response.status == 200:
                    image_data = await response.read()
                    image = Image.open(io.BytesIO(image_data))
                    if max(image.size) > 200 :
                        img_name = f"{uuid4()}.jpg"
                        image = image.save(image_save_dir / img_name)
                        temp = {
                            "image_id" : img_name,
                            "caption" : caption
                        }
                        return temp
                    else :
                        return
                return 
            
    except asyncio.TimeoutError:
        if retries < 1 :
            #await asyncio.sleep(3)
            temp = await download_image(session, semaphore, x, retries + 1) 
            return temp
        return
            
    except :
        return

    
async def main(urls, captions):


    semaphore = asyncio.Semaphore(100)

    async with aiohttp.ClientSession(headers= headers) as session:

        tasks = [download_image(session, semaphore, (url, caption)) for url,caption in zip(urls, captions)]
        captions_list = [await f for f in tqdm(asyncio.as_completed(tasks), total= len(tasks))]
        
        return captions_list

if __name__ ==  "__main__" : 

    if image_save_dir.exists():
        bool = input("Are you Sure? (True/False)")
        if bool == "True":
            shutil.rmtree(image_save_dir)
        else : 
            print("Aborting!..")
            exit()

    image_save_dir.mkdir(parents= True, exist_ok= False)

    data = pd.read_parquet(parquet_file_path, columns=["URL", "top_caption"])
    urls, captions = data["URL"].to_list(), data["top_caption"].to_list()

    urls, captions = urls[:1000000], captions[:1000000]

    for i in range(math.ceil(len(urls)/image_per_batch)) :
        s = i * image_per_batch
        e = min((i+1) * image_per_batch, len(urls))
        print(f"Part{i} of {math.ceil(len(urls)/image_per_batch)}")
        try : 
            output = asyncio.run(main(urls[s:e], captions[s:e]))
            
            captions_list = []
            for x in output:
                if x != None :
                    captions_list.append(x)
            
            caption_save_path = image_save_dir.parent / "captions.json"
    
            if caption_save_path.exists():
                caption_save_path = caption_save_path.parent / f"caption{i}.json"

            with open(caption_save_path, "w") as f : 
                json.dump(captions_list, f, indent= True)

        except KeyboardInterrupt as e:
            print(e)
            exit()
    print("Done ! :)")