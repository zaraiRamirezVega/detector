import cv2
import numpy as np

# --------------- READ DNN MODEL ---------------
config = "deteccionObjetos/model/yolov3.cfg"
weights = "deteccionObjetos/model/yolov3.weights"
LABELS = open("deteccionObjetos/model/coco.names").read().split("\n")
colors = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Load model
net = cv2.dnn.readNetFromDarknet(config, weights)

# Open the camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

while True:
    ret, image = cap.read()
    if not ret:
        print("Error: No se pudo capturar el frame.")
        break

    height, width, _ = image.shape

    # Create a blob
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

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
        for i in idx.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = colors[classIDs[i]].tolist()
            text = "{}: {:.3f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Muestra la imagen con detección
    cv2.imshow("Detección en tiempo real", image)

    # Si se presiona 'q', se cierra la ventana
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra las ventanas
cap.release()
cv2.destroyAllWindows()
