from flask import Flask, request, render_template, send_from_directory
import torch
import cv2
import numpy as np
from utils.tokenizer import Tokenizer
from utils.general import resume_checkpoint
from utils.process_image import letterbox
from pathlib import Path
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

app = Flask(__name__, static_folder="static/")
vocab_path = "vocab.json"
model_path = "saved_model/best.pt"

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No image file provided"

    image = request.files['image']

    if image.filename == '':
        return render_template("upload.html", caption = "No selected file")

    if image:
        file_bytes = np.frombuffer(image.read(), np.uint8)
        img = cv2.imdecode(file_bytes,  cv2.IMREAD_COLOR)
        img = letterbox(img, (480,480))
        img_in = torch.from_numpy(img).to(device).unsqueeze(0).permute(0, 3, 1, 2).contiguous().float()/255
        pred = model.generate(img_in, tokenizer, device=device, greedy= True, top_k=5)
        caption = tokenizer.decode(pred)
        return render_template("upload.html", caption = caption[0])

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if Path("saved_model/best.pt").exists() == False :
        Path("saved_model/").mkdir(parents= True, exist_ok= True)
        os.system("curl -L 'https://drive.google.com/uc?export=download&id=1sc1l1AWDHsKsV_N9OpQcnjGfgwSwdggA&confirm=t' > saved_model/best.pt")

    tokenizer = Tokenizer(vocab_path)
    model = resume_checkpoint(model_path, tokenizer, device=device)
    model.eval()

    app.run(host='0.0.0.0', debug=False)