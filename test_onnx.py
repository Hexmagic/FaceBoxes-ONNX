import cv2
import numpy as np

img = cv2.imread("data/test.jpg")
img = cv2.resize(img,None,None,2.5,2.5,cv2.INTER_LINEAR);
img = np.float32(img)
net = cv2.dnn.readNetFromONNX("FaceBoxes.onnx")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)

blobImage = cv2.dnn.blobFromImage(img,1.0,(img.shape[1],img.shape[0]),(104, 117, 123),False,False)
outNames = net.getUnconnectedOutLayersNames()
net.setInput(blobImage)
outs = net.forward(outNames)
loc,conf = outs
print(loc.shape)
boxes = loc[0].tolist()
scores = conf[0].tolist()
with open('pyout.csv', 'w') as f:
    for i in range(len(boxes)):
        if (scores[i][1]<0.99):
            continue
        print(scores[i][1])
        line = f'{i},{boxes[i][0]:.4f},{boxes[i][1]:.4f},{boxes[i][2]:.4f},{boxes[i][3]:.4f},{scores[i][1]:.6f}\n'
        f.write(line)
