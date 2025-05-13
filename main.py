 
import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet('yolov5.weights', 'yolov5.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load shelf image
image = cv2.imread('shelf.jpg')
height, width, channels = image.shape

# Preprocess image
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Process detections
boxes, confidences, class_ids = [], [], []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # Confidence threshold
            center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-max suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw bounding boxes
for i in indices:
    x, y, w, h = boxes[i[0]]
    label = str(classes[class_ids[i[0]]])
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display result
cv2.imshow('Shelf Monitoring', image)
cv2.waitKey(0)
cv2.destroyAllWindows()