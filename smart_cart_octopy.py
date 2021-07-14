import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
import os
from config import files
from config import paths
import time
# Este biblioteca requiere de:  
# sudo apt-get update -y
# sudo apt-get install -y tesseract-ocr     
import pytesseract
from pytesseract import Output

"""
import easyocr

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=520)])
    except RuntimeError as e:
        print(e)
"""

# Esta linea es para instalar pzbar en el s.o. ya instalada se puede comentar
"""
os.system("sudo apt-get install libzbar0")
os.system("sudo apt install gcc")
os.system("pip install pyzbar")
"""

os.system("v4l2-ctl -d /dev/video0 --list-ctrls")
os.system("v4l2-ctl -d /dev/video0 --set-ctrl=focus_auto=1")
#os.system("v4l2-ctl -d /dev/video0 --set-ctrl=focus_absolute=70")
os.system("v4l2-ctl -d /dev/video0 --list-ctrls")


from pyzbar import pyzbar

# Estas lineas permiten instalar TensorFlow Object Detection API, una vez instalado se pueden comentar
"""
if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
    models_path = paths['APIMODEL_PATH']
    os.system(f'git clone https://github.com/tensorflow/models {models_path}')

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


# Se cargan los archivos para generar el modelo de detección 
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-10')).expect_partial()

# Se define una función para el proceso de detección del código de barras
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return prediction_dict, detections

# Se utiliza el for para buscar la primer cámara disponible, en un sistema de multiples cámaras
for i in range(20):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # 1920
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # 1080 
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width, height)
    if width>0:
        break

# Creación de las categorias
category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

#Entrenamiento del modelo de detección de digitos.
samples = np.loadtxt('generalsamples.data', np.float32)
responses = np.loadtxt('generalresponses.data', np.float32)
responses = responses.reshape((responses.size, 1))
model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

# Configuración para OCR
custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789'

# Erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

# Escala de grises
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Binarización
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

cont_detec = 0
t0 = time.time()
# Lectura continua de la cámara
while cap.isOpened(): 
    ret, frame = cap.read()
    image_np = np.array(frame)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    
    # Detección de los códigos de barras
    prediction_dict, detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # Decodificación de las detecciones
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
        ancho_img = ymax - ymin
        largo_img = xmax - xmin
        box_loc = [xmin*width, xmax*width, ymin*height, ymax*height]
        ancho_img = box_loc[3] - box_loc[2]
        largo_img = box_loc[1] - box_loc[0]
        box_loc = [box_loc[0]-largo_img*0.03, box_loc[1]+largo_img*0.03, ymin*height-ancho_img*0.05, ymax*height+ancho_img*0.05]
        for valor in box_loc:
            if valor <= 0:
                box_loc = [xmin*width, xmax*width, ymin*height, ymax*height]
        box_loc_r = [round(loc) for loc in box_loc]
        coor_boxes.append(box_loc_r)

    # Permite recortar el código de barras de la imagen
    for coor in coor_boxes:
        crop_image = frame[coor[2]:coor[3], coor[0]:coor[1]]
        crop_for_ocr = frame[coor[2]+int(ancho_img*0.80):coor[3], coor[0]:coor[1]]
        #cv2.imshow('Codigo detectado', crop_for_ocr)

        # Procesamiento para el OCR
        ocr_gris = get_grayscale(crop_image)
        ocr_bin = thresholding(ocr_gris)

        output = cv2.connectedComponentsWithStats(ocr_bin, 8, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        mask = np.zeros(ocr_gris.shape, dtype="uint8")

        for i in range(1, numLabels):
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            #print("Elemento", i, "tiene un area de", area)
            #print("Elemento", i, "tiene un largo de ", w)
            keepHeight = h < 50 #and h < 65
            keepWidth = w < 50 #and w < 50

            if all((keepHeight, keepWidth)):
                #print("[INFO] keeping connected component '{}'".format(i))
                componentMask = (labels == i).astype("uint8") * 255
                mask = cv2.bitwise_or(mask, componentMask)
            
        cv2.imshow("Image", crop_image)
        print("Tamaño de la mascara", mask.shape)
        print("Tamaño del recorte", crop_image.shape)
        print("Las coordenadas", coor)
        #crop_for_ocr = mask[coor[2]+int(ancho_img*0.80):coor[3], coor[0]:coor[1]]
        #cv2.imshow("Characters", crop_for_ocr)
        #codigo_ocr = pytesseract.image_to_string(crop_for_ocr, config=custom_config)
        #print('El código es: ',codigo_ocr)

        """
        if len(codigo_ocr) >= 15:
            print(codigo_ocr.split(), len(codigo_ocr))
            if len(codigo_ocr) > 17:
                print(codigo_ocr[1:13])
            else:
                print(codigo_ocr[0:13])
        """
        # Guardar imagen
        # Filename
        filename = 'codigo'+str(cont_detec)+'.jpg'
        #cv2.imwrite(filename, crop_image)


        #Codigo de barras
        barcodes = pyzbar.decode(crop_image)
        for barcode in barcodes:
            code_value = barcode.data.decode("utf-8")
            #print(f"El código detectado es {code_value}")
            cont_detec += 1
            #print(f"El número codigos detectados es: {cont_detec}")
        
        #OCR
        #lect = easyocr.Reader(['en'])
        #numero_barras = lect.readtext(crop_for_ocr)
        #print(numero_barras)

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
    cv2.imshow('Deteccion del codigo', imagen)
    t1 = time.time()
    tiempo = t1 - t0
    #print(f'El tiempo es {tiempo}')
    frames = 1/tiempo
    t0 = t1
    #print(f'Frames por segundo: {frames}')
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break