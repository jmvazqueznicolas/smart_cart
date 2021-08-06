#!/usr/bin/env python3

from PIL import Image
from feature_extractor_VGG16 import FeatureExtractor
from pathlib import Path
import numpy as np
import os
import cv2
import re

if __name__ == '__main__':
    fe = FeatureExtractor()
    path = "./static/img"
    files = os.listdir( path )
    files.remove(".gitkeep")
    files.sort()

    for img_path in files:
        curr_img_path = os.path.join(path, img_path)
        frame = cv2.imread(curr_img_path)
        frame = cv2.resize(frame, (224,224))
        #cv2.imshow("frame",frame)
        #cv2.waitKey(3)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_np = np.array(frame)


        img_path = img_path.rsplit('.',1)
        feature = fe.extract(image_np)
        feature_filename = img_path[0] + ".npy"
        feature_path = os.path.join("./static/feature/VGG16", feature_filename)
        np.save(feature_path, feature)
