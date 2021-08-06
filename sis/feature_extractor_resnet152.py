#!/usr/bin/env python3

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import ResNet152, preprocess_input
from tensorflow.keras.models import Model
import numpy as np

# See https://keras.io/api/applications/ for details

class FeatureExtractor:
    def __init__(self):
        base_model = ResNet152(weights='imagenet')
        #self.base_model = ResNet152(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

    def extract(self, img):
        """
        Extract a deep feature from an input image
        Args:
            img: from PIL.Image.open(path) or tensorflow.keras.preprocessing.image.load_img(path)
            # img must be a np.array of size (224,224,3) in RGB colorspace
        Returns:
            feature (np.ndarray): deep feature with the shape=(4096, )
        """
        #x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(img, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel
        feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
        #self.model.summary()
        return feature / np.linalg.norm(feature)  # Normalize

