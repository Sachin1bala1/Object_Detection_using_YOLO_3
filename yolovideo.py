import cv2
import numpy as np
from matplotlib import pyplot as plt

# load the COCO class labels and yolo model. Get the output layer names
class_names = open("coco.names").read().strip().split("\n")
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
layer_names = net.getLayerNames()
layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

capture = cv2.VideoCapture("NMMC_Route.mp4")
while capture.isOpened():
    ret, image = capture.read()
    (H, W, C) = image.shape
    
    if ret is True:
        # Create the blob with a size of (416, 416), swap red and blue channels, apply scale factor of 1/255
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Feed the input blob to the network, perform inference and get the output:
        net.setInput(blob)
        layerOutputs = net.forward(layer_names)

        # Get inference time:
        t, _ = net.getPerfProfile()
        print('Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency()))

        # Initialization:
        boxes = []
        confidences = []
        class_ids = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # Get class ID and confidence of the current detection:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Filter out weak predictions:
                if confidence > 0.25:
                    # Scale the bounding box coordinates (center, width, height) using the dimensions of the original image:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # Calculate the top-left corner of the bounding box:
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # Update the information we have for each detection:
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # We can apply non-maxima suppression (eliminate weak and overlapping bounding boxes):
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        # Show the results (if any object is detected after non-maxima suppression):
        if len(indices) > 0:
            for i in indices.flatten():
                # Extract the (previously recalculated) bounding box coordinates:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # Draw label and confidence:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = "{}: {:.4f}".format(class_names[class_ids[i]], confidences[i])
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                y = max(y, labelSize[1])
                cv2.rectangle(image, (x, y - labelSize[1]), (x + labelSize[0], y + 0), (0, 255, 0), cv2.FILLED)
                cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Create the dimensions of the figure and set title:
        fig = plt.figure(figsize=(14, 8))
        plt.suptitle("Object detection using OpenCV DNN module and YOLO V3", fontsize=14, fontweight='bold')
        fig.patch.set_facecolor('silver')

        # Plot the results
        img_RGB = image[:, :, ::-1]
        ax = plt.subplot(1, 1, 1)
        
        cv2.imshow('Frame', img_RGB)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        break
        
capture.release()
cv2.destroyAllWindows()

"""
@misc{redmon2018yolov3,
  abstract = {We present some updates to YOLO! We made a bunch of little design changes to
make it better. We also trained this new network that's pretty swell. It's a
little bigger than last time but more accurate. It's still fast though, don't
worry. At 320x320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but
three times faster. When we look at the old .5 IOU mAP detection metric YOLOv3
is quite good. It achieves 57.9 mAP@50 in 51 ms on a Titan X, compared to 57.5
mAP@50 in 198 ms by RetinaNet, similar performance but 3.8x faster. As always,
all the code is online at https://pjreddie.com/yolo/},
  added-at = {2018-09-21T15:24:42.000+0200},
  author = {Redmon, Joseph and Farhadi, Ali},
  biburl = {https://www.bibsonomy.org/bibtex/2cd221fb00896b020dec735d19ff2d9bd/tgandor},
  description = {YOLOv3: An Incremental Improvement},
  interhash = {bbdec3df168e9809d9e61423d4b4e062},
  intrahash = {cd221fb00896b020dec735d19ff2d9bd},
  keywords = {cnn deep_learning object_detection yolo},
  note = {cite arxiv:1804.02767Comment: Tech Report},
  timestamp = {2018-09-21T15:24:42.000+0200},
  title = {YOLOv3: An Incremental Improvement},
  url = {http://arxiv.org/abs/1804.02767},
  year = 2018
}
"""

