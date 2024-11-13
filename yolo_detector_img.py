import cv2
import numpy as np
import os

# --------------- READ DNN MODEL ---------------
# Model configuration
config = "deteccionObjetos/model/yolov3.cfg"
# Weights
weights = "deteccionObjetos/model/yolov3.weights"
# Labels
LABELS = open("deteccionObjetos/model/coco.names").read().split("\n")
colors = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Load model
net = cv2.dnn.readNetFromDarknet(config, weights)

# --------------- READ THE IMAGE AND PREPROCESSING ---------------
# Ruta de la imagen
image_path = "deteccionObjetos/Imagen/imagen_0004.jpg"

# Verifica que la imagen se cargue correctamente
image = cv2.imread(image_path)

if image is None:
    print("Error al cargar la imagen. Verifica la ruta del archivo.")
    exit()  # Salir si la imagen no se pudo cargar

height, width, _ = image.shape

# Create a blob
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                              swapRB=True, crop=False)

# --------------- DETECTIONS AND PREDICTIONS ---------------
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

net.setInput(blob)
outputs = net.forward(ln)

boxes = []
confidences = []
classIDs = []

for output in outputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        if confidence > 0.5:
            box = detection[:4] * np.array([width, height, width, height])
            (x_center, y_center, w, h) = box.astype("int")
            x = int(x_center - (w / 2))
            y = int(y_center - (h / 2))

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            classIDs.append(classID)

idx = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

if len(idx) > 0:
    for i in idx.flatten():  # Flatten the index array to avoid error
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        color = colors[classIDs[i]].tolist()
        text = "{}: {:.3f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                     0.5, color, 2)

# Create output directory if it doesn't exist
output_dir = "deteccionObjetos/ImagenSalida"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Obtener el nombre del archivo original sin la extensión
file_name = os.path.splitext(os.path.basename(image_path))[0]

# Crear el nuevo nombre con el sufijo "_output"
output_image_name = file_name + "_output.jpg"

# Guardar la imagen procesada
output_image_path = os.path.join(output_dir, output_image_name)
cv2.imwrite(output_image_path, image)
print(f"Imagen guardada en: {output_image_path}")

# Display the image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
