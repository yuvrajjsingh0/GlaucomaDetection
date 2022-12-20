from flask import *
import os
import cv2
import keras
import torch
import numpy as np

import math

app = Flask(__name__)

pt_model = torch.hub.load("ultralytics/yolov5", "custom", "best.pt")
def get_roi(img_path, model):
  result = model(img_path)
  bbox = result.xyxy[0][0].detach().cpu().numpy()
  coordinates = [int(i) for i in bbox]
  img = cv2.imread(img_path)
  val = img[coordinates[1]:coordinates[3], coordinates[0]:coordinates[2]]
  return val


print(type(pt_model))
@app.route("/")
def home():
    return render_template("index.html", pred="We are Nightowls!", desc="We ease the process of Glaucoma detection by making the use of Artificial Neural Networks, we also predict the possibility of the same in near furture.")

@app.route("/detect", methods=["POST"])
def upload():
    f = request.files["file"]
    f.save(os.path.join("./STATIC",f.filename))
    img = cv2.imread("./STATIC/"+f.filename)
    print(type(img))
    try:
        new_image = get_roi("./STATIC/"+f.filename, pt_model)
    except:
        new_image = img
    model = keras.models.load_model("glaucoma_2700_epochs.h5")
    img_ = cv2.resize(new_image, (64, 64))
    pred = str(math.ceil(model.predict(img_.reshape(1, 64, 64, 3))[0][0] *100))
    return render_template("index.html", pred=pred, desc="<h5>Precautions:</h5><ul><li>If you are in a high-risk group, get a comprehensive dilated eye exam to catch glaucoma early and start treatment. Prescription eye drops can stop glaucoma from progressing. Your eye care specialist will recommend how often to return for follow-up exams.</li><li>Even if you are not in a high-risk group, getting a comprehensive dilated eye exam by the age of 40 can help catch glaucoma and other eye diseases early.</li><li>Open-angle glaucoma does not have symptoms and is hereditary, so talk to your family members about their vision health to help protect your eyes—and theirs.</li><li>Maintaining a healthy weight, controlling your blood pressure, being physically active, and avoiding smoking will help you avoid vision loss from glaucoma. These healthy behaviors will also help prevent type 2 diabetes and other chronic conditions.</li></ul>")


if __name__=="__main__":
    app.run(debug=True)