from __future__ import division

from models import *
from utils.utils import *
import cv2
import os
import sys
import time
import datetime
import argparse
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from flask import Flask, request, jsonify

app = Flask(__name__)


def detector(img):

    device = torch.device("cpu")

    model = Darknet("config/yolov3.cfg", img_size=416).to(device)

    model.load_darknet_weights("weights/yolov3.weights")

    model.eval()

    classes = load_classes("data/coco.names")
  #  print("*************Request handled*************************")

 #   print("*************Request handled*************************")

    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor()
    ])

    input_imgs = Variable(transform(img).unsqueeze(dim=0)
                          ).type(torch.FloatTensor)
   # print(input_imgs.shape)
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, 0.8, 0.4)

   # print("****Here after model1")
   # cmap = plt.get_cmap("tab20b")
   # colors = [cmap(i) for i in np.linspace(0, 1, 20)]

 #   print("****Here after model2")
    img = np.asarray(img)
    plt.figure()
  #  print("****Here after model3")
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    return_s = {}
    
    if not detections == [None]:

        detections = rescale_boxes(detections[0], 416, img.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
      #  bbox_colors = random.sample(colors, n_cls_preds)
        counter = 0
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            if(cls_conf.item() < 0.5):
                continue

            counter += 1
            print("\t+ Label: %s, Conf: %.5f" %
                  (classes[int(cls_pred)], cls_conf.item()))

            box_w = x2 - x1
            box_h = y2 - y1

            return_s[classes[int(cls_pred)]+"-"+str(counter)
                     ] = (x1.item(), y1.item(), box_w.item(), box_h.item())

    return return_s


@app.route("/predict", methods=["POST"])
def predictx():
    res = detector(Image.open(request.files["file"].stream))
    return jsonify(res)


@app.route("/traffic", methods=["POST"])
def find_traffic():
    img = Image.open(request.files["file"].stream)
    res = detector(img)
    data=None
    prediction="0"

    for key in list(res.keys()):
        print(f"Keys is here : {key}")
        if(key.split("-")[0] == "traffic light"):
            bbox = patches.Rectangle(
                (res[key][0], res[key][1]), res[key][2], res[key][3], linewidth=2, facecolor="none")

            fig, ax = plt.subplots()
            im = ax.imshow(img)
            ax.add_patch(bbox)
            im.set_clip_path(bbox)
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            fig.canvas.draw()

            data = np.fromstring(fig.canvas.tostring_rgb(),
                                 dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
           # fig.savefig("12.png",bbox_inches="tight", pad_inches=0.0)
            break

    if(not data is None ):
        data=remove_background(data)
        model = torch.load("./weights/traffic_model.pth",map_location="cpu")
        prediction = predict_image(data, model)

    return jsonify({"Color":prediction})

def predict_image(image, model):
    transformer = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image = Image.fromarray(image)
    image_tensor = transformer(image)
    image_tensor = image_tensor.unsqueeze(0)
    output = model(image_tensor)
    index = output.argmax().item()

    if (index == 0):
        return "green"
    elif (index == 1):
        return "red"
    else:
        return  "0"


def remove_background(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    x,y,w,h = cv2.boundingRect(cnt)
    dst = img[y:y+h, x:x+w]

    return dst



@app.route("/object",methods=["POST"])
def find_object():
    img=Image.open(request.files["file"].stream)
    res=detector(img)
    objs=set()
    for key in list(res.keys()):
        objs.add(key.split("-")[0])
    return jsonify({"objects": list(objs) })




@app.route("/loadx", methods=["GET"])
def loadx():
    return ''' <html>
    <head> </head>
    <body> 
    
    <form action="/object" method="post" enctype="multipart/form-data">
     Select image to upload:
     <input type="file" name="file" id="file">
  <input type="submit" value="Upload Image" name="submit">
    </form>


    </body>


     </html>
    
    '''


if(__name__ == "__main__"):
    app.run()
