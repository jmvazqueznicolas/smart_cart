import threading
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import re
import math
import pytesseract
from scipy import stats as st  
from pytesseract import Output
from pyzbar import pyzbar
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from config_ssd import files
from config_ssd import paths

# Manipulación de hilos
width = 1280
height = 720

frame1 = np.array([])
frame2 = np.array([])
frame3 = np.array([])

flag1 = False
flag2 = False
flag3 = False

detener_hilos = False

def cam_1(w, h):
    global frame1
    global flag1
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w) # 1920
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h) # 1080
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    #Ver si se puede sacar del for
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        ret, frame1 = cap.read()
        flag1 = True
        if detener_hilos == True:
            cap.release()
            break

def cam_2(w, h):
    global frame2
    global flag2
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w) # 1920
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h) # 1080
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    #Ver si se puede sacar del for
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        ret, frame2 = cap.read()
        flag2 = True
        if detener_hilos == True:
            cap.release()
            break     

def cam_3(w, h):
    global frame3
    global flag3
    cap = cv2.VideoCapture(4)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w) # 1920
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h) # 1080
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    #Ver si se puede sacar del for
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        ret, frame3 = cap.read()
        flag3 = True
        if detener_hilos == True:
            cap.release()
            break   
  
cam1 = threading.Thread(target=cam_1, args=(width,height,))
cam2 = threading.Thread(target=cam_2, args=(width,height,))
cam3 = threading.Thread(target=cam_3, args=(width,height,))

cam1.start()
cam2.start()
cam3.start()

# Configuración para OCR
custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
cont_detec = 0
t0 = time.time()

# Diccionario de productos
productos = {'7501088210709':'Cremino', '7501791600682':'Cafe soluble', 
            '725226003504':'Pulparindo', '0025046021499':'Duvalin',
            '7501025405694':'Toallitas', '7501954906644':'Cafe de grano', 
            '7501032923662':'Oust', '7501005196499':'Knorr',
            '261002800003204':'Molida', '658480001101':'Salmas', 
            '7501071308598':'Frijoles', '7501006559033':'ActII',
            '7501025405090':'Cloralex', '744218120913':'Lucky Gummys', 
            '7501017003341':'La Costena Mango', '7501045401195':'El Dorado',
            '706460249279':'Pedigree', '7501039121610':'Nutrioli',
            '7501003340122':'Mayonesa McCormick', '7502271450049':'Cuchara Bambu', 
            '75071295':'Cigarros PallMall', '7502226294292':'Paracetamol',
            '7501013122053':'Jumex Durazno', '80051671':'Nutella', 
            '7501008042946':'Kelloggs Zucaritas', '7501020515350':'Leche Lala'}
carrito = []
producto_detec = {'nombre':'Sin producto', 'codigo':''}

# Creación de las categorias
category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

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

# Funciones básicas de pre-procesamiento
# Escala de grises
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Binarización
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# Dilatacion
def dilate(image):
    kernel = np.ones((2,2),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)

# Función para la detección del productos
def deteccion_producto(crop_image):
    #crop_image = rotacion(crop_image)
    #cv2.imshow('Rotacion', crop_image)

    #Producto no existente
    producto_detec = {'nombre':'Sin producto', 'codigo':''}
    
    # Procesamiento para el OCR
    ocr_gris = get_grayscale(crop_image)
    ocr_bin = thresholding(ocr_gris)
    output = cv2.connectedComponentsWithStats(ocr_bin, 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    mask = np.zeros(ocr_gris.shape, dtype="uint8")
    # Se eliminan las regiones largas y delgadas (lineas del código de barras)
    for i in range(1, numLabels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        keepHeight = h < 50 #and h < 65
        keepWidth = w < 50 #and w < 50
        if all((keepHeight, keepWidth)):
            componentMask = (labels == i).astype("uint8") * 255
            mask = cv2.bitwise_or(mask, componentMask)
    crop_for_ocr = mask[int(0.7*mask.shape[0]):mask.shape[0], 0:mask.shape[1]]
    crop_for_ocr = dilate(crop_for_ocr)
    crop_for_ocr = cv2.threshold(crop_for_ocr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    codigo_ocr = pytesseract.image_to_string(crop_for_ocr, config=custom_config)
    # Código por OCR
    codigo_ocr = re.sub('\D', '', codigo_ocr)
    # Código de barras por Pyzbar
    barcode = pyzbar.decode(crop_image)
    #for barcode in barcodes:
    if barcode:
        code_value = barcode[0].data.decode("utf-8")
    else:
        code_value = ''

    # Producto nuevo?
    if (producto_detec['codigo'] != (code_value or codigo_ocr)) and ((code_value or codigo_ocr) in productos):
        # Se actualiza producto
        try:
            producto_detec = {'nombre':productos[code_value], 'codigo':code_value}
        except:
            producto_detec = {'nombre':productos[codigo_ocr], 'codigo':codigo_ocr}
        nombre_producto = producto_detec['nombre']
        print(nombre_producto)
        time.sleep(2)
        #confirmacion = input(f'Confirma comprar {nombre_producto} ')
        #if confirmacion == "si":
        #    carrito.append(producto_detec['nombre'])
        #    print('El contenido del carrito es: ')
        #    for producto in carrito:
        #        print(producto)
    
    return carrito

# Ciclo principal
terminar = False
while terminar == False:
    # Se pregunta si las cámaras estan listas para ser leidas
    if (flag1 == True) and (flag2 == True) and (flag3 == True):
        frames = [frame1, frame2, frame3]
        # Se procesan los frames    
        for cam, frame in enumerate(frames, 1):
            # Se manda la imagen para detectar el codigo de barras
            image_np = np.array(frame)
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            # Detección de los productos
            prediction_dict, detections = detect_fn(input_tensor)
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # Decodificación de las detecciones
            classes = detections['detection_classes']
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            # Selección de regiones con códigos de barras
            boxes = np.squeeze([detections['detection_boxes']])
            scores = np.squeeze([detections['detection_scores']])
            min_score_tresh = 0.7
            tboxes = boxes[scores > min_score_tresh]
            
            # Para cada una de las ROIs con códigos
            coor_boxes = []
            for box in tboxes:
                ymin, xmin, ymax, xmax = box
                box_loc = [xmin*width, xmax*width, ymin*height, ymax*height]
                box_loc_r = [round(loc) for loc in box_loc]
                coor_boxes.append(box_loc_r)
                
            label, _ = viz_utils.visualize_boxes_and_labels_on_image_array(
                        image_np_with_detections,
                        detections['detection_boxes'],
                        detections['detection_classes']+label_id_offset,
                        detections['detection_scores'],
                        category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=5,
                        min_score_thresh=.7,
                        agnostic_mode=False)
            
            if label != '':
                product = productos[label]
                print(f'Producto detectado: {product}')
            
            
            imagen = cv2.resize(image_np_with_detections, (480, 320))
            
            # Colocando texto en la Imagen
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (350, 500)
            org2 = (350,550)
            fontScale = 1.2
            color = (0, 255, 0)
            color2 = (0, 0, 255)
            thickness = 2
            cv2.putText(imagen, 'Octopy', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(imagen, producto_detec['nombre'], org2, font, 
                        fontScale, color2, thickness, cv2.LINE_AA)
            if cam==1:
                cv2.imshow('Camara 1', imagen)
            elif cam==2:
                cv2.imshow('Camara 2', imagen)
            elif cam==3:
                cv2.imshow('Camara 3', imagen)


            # Images de las camaras (se puede optimizar) 
            #frame1 = cv2.resize(frame1, (480, 320))
            #frame2 = cv2.resize(frame2, (480, 320))
            #frame3 = cv2.resize(frame3, (480, 320))
            #Hori = np.concatenate((frame1, frame2, frame3), axis=1)
            #cv2.imshow('Todas', Hori)

            # Cálculo del tiempo de ejecución
            t1 = time.time()
            tiempo = t1 - t0
            #frames = 1/tiempo
            t0 = t1        
            #print(tiempo)
            #print(frames)
            #cv2.imshow('Camara1', frame1)
            #cv2.imshow('Camara2', frame2)
            #cv2.imshow('Camara3', frame3)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                    detener_hilos = True
                    terminar = True
                    cv2.destroyAllWindows()
                    break