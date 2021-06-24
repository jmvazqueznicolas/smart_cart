import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
import os
from config import files
from config import paths

# Esta linea es para instalar la libreria para el sistema operativo
"""
os.system("sudo apt-get install libzbar0")
"""
from pyzbar import pyzbar

# Estas lineas permiten instalar TensorFlow Object Detection API, una vez instalado se pueden comentar
"""
if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
    models_path = paths['APIMODEL_PATH']
    os.system(f'git clone https://github.com/tensorflow/models {models_path}')

    os.system("sudo apt install gcc")
    os.system("sudo apt  install protobuf-compiler")
    os.system("cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install .")

    VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
    os.system(f"python {VERIFICATION_SCRIPT}")
"""

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

# Para llevar a cabo la ejecución se necesitan algunos archivos
# 1.- Los archivos ckpt-3
# 2.- pipeline.config
# Estos deben de estar en la carpet /Tensorflow/workspace/models/[MODELO_UTILIZADO]/
# Otro archivo necesario es
# 1.- label_map.pbtxt
# El cual debe estar en la carpeta Tensorflow/workspace/annotations/

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return prediction_dict, detections

# Se utiliza el for para buscar la primer cámara disponible, en un sistema de multiples cámaras
for i in range(20):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width>0:
        break

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

print(width, height)
while cap.isOpened(): 
    ret, frame = cap.read()
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    prediction_dict, detections = detect_fn(input_tensor)
    #print('Este es el diccionario de predicciones')
    #print(prediction_dict)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    # Selección de regiones con códigos de barras
    boxes = np.squeeze([detections['detection_boxes']])
    scores = np.squeeze([detections['detection_scores']])
    min_score_tresh = 0.8
    tboxes = boxes[scores > min_score_tresh]
    
    coor_boxes = []
    for box in tboxes:
        ymin, xmin, ymax, xmax = box
        box_loc = [xmin*width, xmax*width, ymin*height, ymax*height]
        box_loc_r = [round(loc) for loc in box_loc]
        coor_boxes.append(box_loc_r)
    
    barcodes = pyzbar.decode(frame)
    if len(barcodes)>0:
        print("Se detecto un código")

    # Permite recortar el código de barras de la imagen
    for coor in coor_boxes:
        crop_image = frame[coor[2]:coor[3], coor[0]:coor[1]]
       # barcodes = pyzbar.decode(crop_image)
       # print(len(barcodes))
        cv2.imshow('Recorte', crop_image)
        for barcode in barcodes:
            code_value = barcode.data.decode("utf-8")
            print(f"El código detectado es {code_value}")

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)
    imagen = cv2.resize(image_np_with_detections, (800, 600))
    cv2.imshow('object detection', imagen)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break