from flask import Flask, request, render_template, send_from_directory
import torch
import cv2
import numpy as np
from utils.tokenizer import Tokenizer
from utils.general import resume_checkpoint
from utils.process_image import letterbox


app = Flask(__name__, static_folder="static/")
vocab_path = "vocab.json"
model_path = "saved_model/epoch1.pt"

# Configuration for file uploads
UPLOAD_FOLDER = 'images/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

tokenizer = Tokenizer(vocab_path)
model = resume_checkpoint(model_path, tokenizer).cuda()
model.eval()

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
        file_bytes = np.fromstring(image.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        img = letterbox(img, (480,480))
        img_in = torch.from_numpy(img).cuda().unsqueeze(0).permute(0, 3, 1, 2).contiguous().float()/255
        pred = model.generate(img_in, tokenizer, greedy= True, top_k=5)
        caption = tokenizer.decode(pred)
        return render_template("upload.html", caption = caption[0])

@app.route('/generate', methods=['POST'])
def aupload_image():
    if 'image' not in request.files:
        return "No image file provided"

    image = request.files['image']

    if image.filename == '':
        return "No selected file"

    if image:
        image.save(f"{app.config['UPLOAD_FOLDER']}/{image.filename}")
        return f"Image uploaded successfully: {image.filename}"

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)