#!/usr/bin/env python3

import numpy as np
import cv2

from PIL import Image
from feature_extractor_resnet152 import FeatureExtractor
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import os
import time


def get_product_name(texto):
    texto = texto.rsplit('/',3)
    texto = texto[3]
    texto = texto.rsplit('_',1)
    return str(texto[0])

def main():
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # 1920
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # 1080 

    # Read image features
    fe = FeatureExtractor()
    features = []
    img_paths = []

    path = "./static/img"
    img_files = os.listdir( path )
    img_files.remove(".gitkeep")

    for feature_path in Path("./static/feature/resnet152").glob("*.npy"):
        features.append(np.load(feature_path))
        for image_file in img_files:
            if image_file.startswith(feature_path.stem):
                curr_img_path = os.path.join(path, image_file)
                img_paths.append(curr_img_path)
    features = np.array(features)

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (20, 50)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2

    num_frames = 60
    curr_frame = 0
    fig = plt.figure()
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        curr_frame += 1
        if (curr_frame == num_frames): curr_frame = 0

        if (curr_frame == 1):
            plt.title("Visual search")
            ax1 = fig.add_subplot(2,2,1)
            ax2 = fig.add_subplot(2,2,2)
            ax3 = fig.add_subplot(2,2,3)
            ax4 = fig.add_subplot(2,2,4)
            ax1.set_title("frame")
            ax2.set_title("img0")
            ax3.set_title("img1")
            ax4.set_title("img2")

            # Run search
            frame = cv2.resize(frame, (224,224))    # Needs an RGB image of 224x224
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_np = np.array(frame)

            query = fe.extract(image_np)
            dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
            ids = np.argsort(dists)[:3]  # Top 3 results
            scores = [(dists[id], img_paths[id]) for id in ids]

            img0 = cv2.imread(str(img_paths[ids[0]]))
            img1 = cv2.imread(str(img_paths[ids[1]]))
            img2 = cv2.imread(str(img_paths[ids[2]]))
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

            product_name0 = get_product_name(scores[0][1])
            product_name1 = get_product_name(scores[1][1])
            product_name2 = get_product_name(scores[2][1])

            if ((scores[0][0] < 0.81) and ((product_name0 == product_name1) or (product_name0   ==  product_name2))): 
                image_np = cv2.putText(image_np, product_name0, org, font, fontScale, color,        thickness, cv2.LINE_AA)


            ax1.imshow(image_np)
            ax2.imshow(img0)
            ax3.imshow(img1)
            ax4.imshow(img2)
            plt.show(block=False)        
            plt.pause(0.00001)
            plt.clf()
           
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()